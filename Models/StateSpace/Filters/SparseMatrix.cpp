// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include <iostream>
#include <utility>
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/DiagonalMatrix.hpp"
#include "Models/StateSpace/Filters/SparseVector.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  void SparseKalmanMatrix::conforms_to_rows(int i) const {
    if (i == nrow()) return;
    std::ostringstream err;
    err << "object of length " << i
        << " does not conform with the number of rows (" << nrow() << ")";
    report_error(err.str());
  }

  void SparseKalmanMatrix::conforms_to_cols(int i) const {
    if (i == ncol()) return;
    std::ostringstream err;
    err << "object of length " << i
        << " does not conform with the number of columns (" << ncol() << ")";
    report_error(err.str());
  }

  namespace {
    template <class VECTOR>
    Vector sparse_multiply_impl(const SparseMatrixBlock &m, const VECTOR &v) {
      m.conforms_to_cols(v.size());
      Vector ans(m.nrow(), 0.0);
      m.multiply(VectorView(ans), v);
      return ans;
    }
  }  // namespace

  Vector SparseMatrixBlock::operator*(const Vector &v) const {
    return sparse_multiply_impl(*this, v);
  }
  Vector SparseMatrixBlock::operator*(const VectorView &v) const {
    return sparse_multiply_impl(*this, v);
  }
  Vector SparseMatrixBlock::operator*(const ConstVectorView &v) const {
    return sparse_multiply_impl(*this, v);
  }
  Matrix SparseMatrixBlock::operator*(const Matrix &rhs) const {
    conforms_to_cols(rhs.nrow());
    Matrix ans(nrow(), rhs.ncol());
    for (int j = 0; j < rhs.ncol(); ++j) {
      multiply(ans.col(j), rhs.col(j));
    }
    return ans;
  }

  Vector SparseMatrixBlock::Tmult(const ConstVectorView &rhs) const {
    conforms_to_rows(rhs.size());
    Vector ans(ncol());
    Tmult(VectorView(ans), rhs);
    return ans;
  }

  Matrix SparseMatrixBlock::Tmult(const Matrix &rhs) const {
    conforms_to_rows(rhs.nrow());
    Matrix ans(ncol(), rhs.ncol());
    for (int j = 0; j < ans.ncol(); ++j) {
      Tmult(ans.col(j), rhs.col(j));
    }
    return ans;
  }
  
  void SparseKalmanMatrix::check_can_add(const SubMatrix &block) const {
    if (block.nrow() != nrow() || block.ncol() != ncol()) {
      std::ostringstream err;
      err << "cant add SparseMatrix to SubMatrix: rows and columnns "
          << "are incompatible" << endl
          << "this->nrow() = " << nrow() << endl
          << "this->ncol() = " << ncol() << endl
          << "that.nrow()  = " << block.nrow() << endl
          << "that.ncol()  = " << block.ncol() << endl;
      report_error(err.str());
    }
  }

  void SparseMatrixBlock::matrix_multiply_inplace(SubMatrix m) const {
    for (int i = 0; i < m.ncol(); ++i) {
      multiply_inplace(m.col(i));
    }
  }

  Matrix SparseMatrixBlock::dense() const {
    if (nrow() == ncol()) {
      Matrix ans(nrow(), ncol(), 0.0);
      ans.diag() = 1.0;
      matrix_multiply_inplace(SubMatrix(ans));
      return ans;
    } else {
      return *this * SpdMatrix(ncol(), 1.0);
    }
  }
  
  void SparseMatrixBlock::matrix_transpose_premultiply_inplace(
      SubMatrix m) const {
    for (int i = 0; i < m.nrow(); ++i) {
      multiply_inplace(m.row(i));
    }
  }

  void SparseMatrixBlock::left_inverse(VectorView,
                                       const ConstVectorView &) const {
    report_error("left_inverse was called for a sparse matrix type that is "
                 "either mathematically non-invertible, or the left_inverse "
                 "function simply has not been implemented.");
  }

  Matrix & SparseMatrixBlock::add_to(Matrix &P) const {
    add_to_block(SubMatrix(P));
    return P;
  }
  
  Matrix SparseKalmanMatrix::dense() const {
    Matrix ans(nrow(), ncol(), 0.0);
    add_to(ans);
    return ans;
  }

  //======================================================================
  BlockDiagonalMatrixBlock::BlockDiagonalMatrixBlock(
      const BlockDiagonalMatrixBlock &rhs)
      : dim_(0) {
    for (int i = 0; i < rhs.blocks_.size(); ++i) {
      add_block(rhs.blocks_[i]->clone());
    }
  }

  BlockDiagonalMatrixBlock *BlockDiagonalMatrixBlock::clone() const {
    return new BlockDiagonalMatrixBlock(*this);
  }

  BlockDiagonalMatrixBlock &BlockDiagonalMatrixBlock::operator=(
      const BlockDiagonalMatrixBlock &rhs) {
    if (this != &rhs) {
      blocks_.clear();
      for (int i = 0; i < rhs.blocks_.size(); ++i) {
        add_block(rhs.blocks_[i]->clone());
      }
    }
    return *this;
  }

  void BlockDiagonalMatrixBlock::add_block(
      const Ptr<SparseMatrixBlock> &block) {
    if (!block) {
      report_error("nullptr argument passed to BlockDiagonalMatrixBlock::"
                   "add_block");
    }
    if (block->nrow() != block->ncol()) {
      report_error("Sub-blocks of a BlockDiagonalMatrixBlock must be square.");
    }
    dim_ += block->nrow();
    blocks_.push_back(block);
  }

  void BlockDiagonalMatrixBlock::check_can_multiply(
      const VectorView &lhs, const ConstVectorView &rhs) const {
    if (lhs.size() != dim_) {
      report_error("Left hand side is the wrong dimension.");
    }
    if (rhs.size() != dim_) {
      report_error("Right hand side is the wrong dimension.");
    }
  }

  void BlockDiagonalMatrixBlock::multiply(VectorView lhs,
                                          const ConstVectorView &rhs) const {
    check_can_multiply(lhs, rhs);
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->nrow();
      VectorView left(lhs, position, local_dim);
      ConstVectorView right(rhs, position, local_dim);
      blocks_[b]->multiply(left, right);
      position += local_dim;
    }
  }

  void BlockDiagonalMatrixBlock::multiply_and_add(
      VectorView lhs, const ConstVectorView &rhs) const {
    check_can_multiply(lhs, rhs);
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->nrow();
      VectorView left(lhs, position, local_dim);
      ConstVectorView right(rhs, position, local_dim);
      blocks_[b]->multiply_and_add(left, right);
      position += local_dim;
    }
  }

  void BlockDiagonalMatrixBlock::Tmult(VectorView lhs,
                                       const ConstVectorView &rhs) const {
    check_can_multiply(lhs, rhs);
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->nrow();
      VectorView left(lhs, position, local_dim);
      ConstVectorView right(rhs, position, local_dim);
      blocks_[b]->Tmult(left, right);
      position += local_dim;
    }
  }

  void BlockDiagonalMatrixBlock::multiply_inplace(VectorView x) const {
    conforms_to_cols(x.size());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->nrow();
      VectorView local(x, position, local_dim);
      blocks_[b]->multiply_inplace(local);
      position += local_dim;
    }
  }

  void BlockDiagonalMatrixBlock::matrix_multiply_inplace(SubMatrix m) const {
    conforms_to_cols(m.nrow());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->nrow();
      SubMatrix rows_of_m(m, position, position + local_dim - 1, 0,
                          m.ncol() - 1);
      blocks_[b]->matrix_multiply_inplace(rows_of_m);
      position += local_dim;
    }
  }

  void BlockDiagonalMatrixBlock::matrix_transpose_premultiply_inplace(
      SubMatrix m) const {
    // The number of columns in m must match the number of rows in
    // this->transpose(), which is the same as the number of rows in this.
    conforms_to_cols(m.ncol());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      // We really want the number of rows in blocks_[b].transpose(), but since
      // the matrices are square it does not matter.
      int local_dim = blocks_[b]->ncol();
      SubMatrix m_columns(m, 0, m.nrow() - 1, position,
                          position + local_dim - 1);
      blocks_[b]->matrix_transpose_premultiply_inplace(m_columns);
      position += local_dim;
    }
  }

  SpdMatrix BlockDiagonalMatrixBlock::inner() const {
    SpdMatrix ans(ncol());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->ncol();
      SubMatrix inner_block(ans, position, position + local_dim - 1,
                            position, position + local_dim - 1);
      inner_block = blocks_[b]->inner();
      position += local_dim;
    }
    return ans;
  }

  SpdMatrix BlockDiagonalMatrixBlock::inner(const ConstVectorView &weights) const {
    SpdMatrix ans(ncol());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->ncol();
      const ConstVectorView local_weights(weights, position, local_dim);
      SubMatrix inner_block(ans, position, position + local_dim - 1,
                            position, position + local_dim - 1);
      inner_block = blocks_[b]->inner(local_weights);
      position += local_dim;
    }
    return ans;
  }
  
  void BlockDiagonalMatrixBlock::add_to_block(SubMatrix block) const {
    conforms_to_rows(block.nrow());
    conforms_to_cols(block.ncol());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->nrow();
      SubMatrix local(block, position, position + local_dim - 1, position,
                      position + local_dim - 1);
      blocks_[b]->add_to_block(local);
      position += local_dim;
    }
  }

  //======================================================================
  StackedMatrixBlock::StackedMatrixBlock(const StackedMatrixBlock &rhs)
      : nrow_(0), ncol_(0) {
    for (int b = 0; b < rhs.blocks_.size(); ++b) {
      add_block(rhs.blocks_[b]->clone());
    }
  }

  StackedMatrixBlock &StackedMatrixBlock::operator=(
      const StackedMatrixBlock &rhs) {
    if (&rhs != this) {
      nrow_ = 0;
      ncol_ = 0;
      blocks_.clear();
      for (int b = 0; b < rhs.blocks_.size(); ++b) {
        add_block(rhs.blocks_[b]->clone());
      }
    }
    return *this;
  }

  void StackedMatrixBlock::clear() {
    blocks_.clear();
    nrow_ = 0;
    ncol_ = 0;
  }
  
  void StackedMatrixBlock::add_block(const Ptr<SparseMatrixBlock> &block) {
    if (nrow_ == 0) {
      nrow_ = block->nrow();
      ncol_ = block->ncol();
    } else {
      if (block->ncol() != ncol_) {
        report_error(
            "Blocks in a stacked matrix must have the same "
            "number of columns.");
      }
      nrow_ += block->nrow();
    }
    blocks_.push_back(block);
  }

  void StackedMatrixBlock::multiply(VectorView lhs,
                                    const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int nr = blocks_[b]->nrow();
      VectorView view(lhs, position, nr);
      blocks_[b]->multiply(view, rhs);
      position += nr;
    }
  }

  void StackedMatrixBlock::multiply_and_add(VectorView lhs,
                                            const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int nr = blocks_[b]->nrow();
      VectorView view(lhs, position, nr);
      blocks_[b]->multiply_and_add(view, rhs);
      position += nr;
    }
  }

  void StackedMatrixBlock::Tmult(VectorView lhs,
                                 const ConstVectorView &rhs) const {
    conforms_to_cols(lhs.size());
    conforms_to_rows(rhs.size());
    int position = 0;
    lhs = 0;
    Vector workspace(ncol_, 0.0);
    for (int b = 0; b < blocks_.size(); ++b) {
      int stride = blocks_[b]->nrow();
      ConstVectorView view(rhs, position, stride);
      blocks_[b]->Tmult(VectorView(workspace), view);
      lhs += workspace;
      position += stride;
    }
  }

  void StackedMatrixBlock::multiply_inplace(VectorView x) const {
    report_error("multiply_inplace only works for square matrices.");
  }

  SpdMatrix StackedMatrixBlock::inner() const {
    SpdMatrix ans(ncol(), 0.0);
    for (int b = 0; b < blocks_.size(); ++b) {
      ans += blocks_[b]->inner();
    }
    return ans;
  }

  SpdMatrix StackedMatrixBlock::inner(const ConstVectorView &weights) const {
    if (weights.size() != nrow()) {
      report_error("Weight vector was the wrong size.");
    }
    SpdMatrix ans(ncol(), 0.0);
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->nrow();
      const ConstVectorView local_weights(weights, position, local_dim);
      ans += blocks_[b]->inner(local_weights);
      position += local_dim;
    }
    return ans;
  }
  
  void StackedMatrixBlock::add_to_block(SubMatrix block) const {
    conforms_to_rows(block.nrow());
    conforms_to_cols(block.ncol());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      SubMatrix lhs_block(block, position, position + blocks_[b]->nrow() - 1,
                          0, ncol_ - 1);
      blocks_[b]->add_to_block(lhs_block);
      position += blocks_[b]->nrow();
    }
  }

  Matrix StackedMatrixBlock::dense() const {
    Matrix ans(nrow(), ncol());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      SubMatrix ans_block(ans, position, position + blocks_[b]->nrow() - 1,
                          0, ncol_ - 1);
      ans_block = blocks_[b]->dense();
      position += blocks_[b]->nrow();
    }
    return ans;
  }
  
  //======================================================================
  LocalLinearTrendMatrix *LocalLinearTrendMatrix::clone() const {
    return new LocalLinearTrendMatrix(*this);
  }

  void LocalLinearTrendMatrix::multiply(VectorView lhs,
                                        const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    lhs[0] = rhs[0] + rhs[1];
    lhs[1] = rhs[1];
  }

  void LocalLinearTrendMatrix::multiply_and_add(
      VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    lhs[0] += rhs[0] + rhs[1];
    lhs[1] += rhs[1];
  }

  void LocalLinearTrendMatrix::Tmult(VectorView lhs,
                                     const ConstVectorView &rhs) const {
    conforms_to_cols(lhs.size());
    conforms_to_rows(rhs.size());
    lhs[0] = rhs[0];
    lhs[1] = rhs[0] + rhs[1];
  }

  void LocalLinearTrendMatrix::multiply_inplace(VectorView v) const {
    conforms_to_cols(v.size());
    v[0] += v[1];
  }

  SpdMatrix LocalLinearTrendMatrix::inner() const {
    // 1 0 * 1 1  = 1 1
    // 1 1   0 1    1 2
    SpdMatrix ans(2);
    ans = 1.0;
    ans(1, 1) = 2.0;
    return ans;
  }

  SpdMatrix LocalLinearTrendMatrix::inner(const ConstVectorView &weights) const {
    // 1 0 * w1 0  * 1 1  =  w1  w1
    // 1 1   0  w2   0 1     w1  w1 + w2 
    if (weights.size() != 2) {
      report_error("Wrong size weight vector");
    }
    SpdMatrix ans(2);
    ans(0, 0) = ans(0, 1) = ans(1, 0) = weights[0];
    ans(1, 1) = weights[0] + weights[1];
    return ans;
  }
  
  void LocalLinearTrendMatrix::add_to_block(SubMatrix block) const {
    check_can_add(block);
    block.row(0) += 1;
    block(1, 1) += 1;
  }

  Matrix LocalLinearTrendMatrix::dense() const {
    Matrix ans(2, 2, 1.0);
    ans(1, 0) = 0.0;
    return ans;
  }

  //======================================================================
  namespace {
    typedef DiagonalMatrixParamView DMPV;
  }  // namespace

  void DMPV::add_variance(const Ptr<UnivParams> &variance) {
    variances_.push_back(variance);
    set_observer(variance);
    current_ = false;
  }

  void DMPV::ensure_current() const {
    if (current_) return;
    diagonal_elements_.resize(variances_.size());
    for (int i = 0; i < diagonal_elements_.size(); ++i) {
      diagonal_elements_[i] = variances_[i]->value();
    }
    current_ = true;
  }

  void DMPV::set_observer(const Ptr<UnivParams> &variance) {
    variance->add_observer([this]() { current_ = false; });
  }

  //======================================================================
  namespace {
    typedef SparseDiagonalMatrixBlockParamView SDMB;
  }  // namespace

  SDMB *SDMB::clone() const { return new SDMB(*this); }

  void SDMB::add_element(const Ptr<UnivParams> &element, int position) {
    if (position < 0) {
      report_error("Position must be non-negative.");
    }
    if (!positions_.empty() && position < positions_.back()) {
      report_error("Please add elements in position order.");
    }
    if (position >= dim_) {
      report_error("Position value exceeds matrix dimension.");
    }
    elements_.push_back(element);
    positions_.push_back(position);
  }

  void SDMB::multiply(VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    lhs = 0;
    for (int i = 0; i < positions_.size(); ++i) {
      int pos = positions_[i];
      lhs[pos] = rhs[pos] * elements_[i]->value();
    }
  }

  void SDMB::multiply_and_add(VectorView lhs,
                              const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    for (int i = 0; i < positions_.size(); ++i) {
      int pos = positions_[i];
      lhs[pos] += rhs[pos] * elements_[i]->value();
    }
  }

  void SDMB::Tmult(VectorView lhs, const ConstVectorView &rhs) const {
    multiply(lhs, rhs);
  }

  void SDMB::multiply_inplace(VectorView x) const {
    conforms_to_cols(x.size());
    int x_index = 0;
    for (int i = 0; i < positions_.size(); ++i) {
      int pos = positions_[i];
      while (x_index < pos) {
        x[x_index++] = 0;
      }
      x[x_index++] *= elements_[i]->value();
    }
    while (x_index < x.size()) {
      x[x_index++] = 0;
    }
  }

  SpdMatrix SDMB::inner() const {
    Matrix ans(nrow(), 0.0);
    for (int i = 0; i < positions_.size(); ++i) {
      int pos = positions_[i];
      ans(pos, pos) = square(elements_[i]->value());
    }
    return ans;
  }

  SpdMatrix SDMB::inner(const ConstVectorView &weights) const {
    if (weights.size() != nrow()) {
      report_error("Wrong size weight vector.");
    }
    Matrix ans(nrow(), 0.0);
    for (int i = 0; i < positions_.size(); ++i) {
      int pos = positions_[i];
      ans(pos, pos) = square(elements_[i]->value()) * weights[i];
    }
    return ans;
  }
  
  void SDMB::add_to_block(SubMatrix block) const {
    conforms_to_cols(block.ncol());
    conforms_to_rows(block.nrow());
    for (int i = 0; i < positions_.size(); ++i) {
      int pos = positions_[i];
      block(pos, pos) += elements_[i]->value();
    }
  }

  //======================================================================
  namespace {
    typedef SeasonalStateSpaceMatrix SSSM;
  }  // namespace

  SSSM::SeasonalStateSpaceMatrix(int number_of_seasons)
      : number_of_seasons_(number_of_seasons) {}

  SeasonalStateSpaceMatrix *SSSM::clone() const {
    return new SeasonalStateSpaceMatrix(*this);
  }

  int SSSM::nrow() const { return number_of_seasons_ - 1; }

  int SSSM::ncol() const { return number_of_seasons_ - 1; }

  void SSSM::multiply(VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    lhs[0] = 0;
    for (int i = 0; i < ncol(); ++i) {
      lhs[0] -= rhs[i];
      if (i > 0) lhs[i] = rhs[i - 1];
    }
  }

  void SSSM::multiply_and_add(VectorView lhs,
                              const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    for (int i = 0; i < ncol(); ++i) {
      lhs[0] -= rhs[i];
      if (i > 0) lhs[i] += rhs[i - 1];
    }
  }

  void SSSM::Tmult(VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(rhs.size());
    conforms_to_cols(lhs.size());
    double first = rhs[0];
    for (int i = 0; i < rhs.size() - 1; ++i) {
      lhs[i] = rhs[i + 1] - first;
    }
    lhs[rhs.size() - 1] = -first;
  }

  void SSSM::multiply_inplace(VectorView x) const {
    conforms_to_rows(x.size());
    int stride = x.stride();
    int n = x.size();
    double *now = &x[n - 1];
    double total = -*now;
    for (int i = 0; i < n - 1; ++i) {
      double *prev = now - stride;
      total -= *prev;
      *now = *prev;
      now = prev;
    }
    *now = total;
  }

  SpdMatrix SSSM::inner() const {
    // -1  1  0  0 .... 0          -1 -1 -1 -1 ... -1
    // -1  0  1  0 .... 0           1  0  0  0 .... 0
    // -1  0  0  1 .... 0           0  1  0  0 .... 0
    // -1  0  0  0 .... 0           0  0  1  0 .... 0
    // -1  0  0  0 .... 0           0  0  0  1 .... 0
    SpdMatrix ans(nrow());
    ans = 1.0;
    ans.diag() = 2.0;
    ans.diag().back() = 1.0;
    return ans;
  }

  SpdMatrix SSSM::inner(const ConstVectorView &weights) const {
    // -1  1  0  0 .... 0   w1        -1 -1 -1 -1 ... -1
    // -1  0  1  0 .... 0     w2       1  0  0  0 .... 0
    // -1  0  0  1 .... 0       w3     0  1  0  0 .... 0
    // -1  0  0  0 .... 0         w4   0  0  1  0 .... 0
    // -1  0  0  0 .... 0           w5 0  0  0  1 .... 0
    //
    // = w1 + w2  w1    w1 ....
    //   w1     w1 + w3 w1 ....
    //   w1       w1    w1 + w4
    if (weights.size() != nrow()) {
      report_error("Wrong size weight vector.");
    }
    SpdMatrix ans(nrow(), 0.0);
    ans += weights[0];
    VectorView(ans.diag(), 0, nrow() - 1) +=
        ConstVectorView(weights, 1, nrow() - 1);
    return ans;
  }
  
  void SSSM::add_to_block(SubMatrix block) const {
    check_can_add(block);
    block.row(0) -= 1;
    VectorView d(block.subdiag(1));
    d += 1;
  }

  Matrix SSSM::dense() const {
    Matrix ans(nrow(), ncol(), 0.0);
    ans.row(0) = -1;
    ans.subdiag(1) = 1.0;
    return ans;
  }
  //======================================================================
  AutoRegressionTransitionMatrix::AutoRegressionTransitionMatrix(
      const Ptr<GlmCoefs> &rho)
      : autoregression_params_(rho) {}

  AutoRegressionTransitionMatrix::AutoRegressionTransitionMatrix(
      const AutoRegressionTransitionMatrix &rhs)
      : SparseMatrixBlock(rhs),
        autoregression_params_(rhs.autoregression_params_->clone()) {}

  AutoRegressionTransitionMatrix *AutoRegressionTransitionMatrix::clone()
      const {
    return new AutoRegressionTransitionMatrix(*this);
  }

  int AutoRegressionTransitionMatrix::nrow() const {
    return autoregression_params_->nvars_possible();
  }

  int AutoRegressionTransitionMatrix::ncol() const {
    return autoregression_params_->nvars_possible();
  }

  void AutoRegressionTransitionMatrix::multiply(
      VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    lhs[0] = 0;
    int p = nrow();
    const Vector &rho(autoregression_params_->value());
    for (int i = 0; i < p; ++i) {
      lhs[0] += rho[i] * rhs[i];
      if (i > 0) lhs[i] = rhs[i - 1];
    }
  }

  void AutoRegressionTransitionMatrix::multiply_and_add(
      VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    int p = nrow();
    const Vector &rho(autoregression_params_->value());
    for (int i = 0; i < p; ++i) {
      lhs[0] += rho[i] * rhs[i];
      if (i > 0) lhs[i] += rhs[i - 1];
    }
  }

  void AutoRegressionTransitionMatrix::Tmult(VectorView lhs,
                                             const ConstVectorView &rhs) const {
    conforms_to_rows(rhs.size());
    conforms_to_cols(lhs.size());
    int p = ncol();
    const Vector &rho(autoregression_params_->value());
    for (int i = 0; i < p; ++i) {
      lhs[i] = rho[i] * rhs[0] + (i + 1 < p ? rhs[i + 1] : 0);
    }
  }

  void AutoRegressionTransitionMatrix::multiply_inplace(VectorView x) const {
    conforms_to_cols(x.size());
    int p = x.size();
    double first_entry = 0;
    const Vector &rho(autoregression_params_->value());
    for (int i = p - 1; i >= 0; --i) {
      first_entry += rho[i] * x[i];
      if (i > 0) {
        x[i] = x[i - 1];
      } else {
        x[i] = first_entry;
      }
    }
  }

  SpdMatrix AutoRegressionTransitionMatrix::inner() const {
    SpdMatrix ans = outer(autoregression_params_->value());
    int dim = ans.nrow();
    VectorView(ans.diag(), 0, dim - 1) += 1.0;
    return ans;
  }

  SpdMatrix AutoRegressionTransitionMatrix::inner(
      const ConstVectorView &weights) const {
    SpdMatrix ans = outer(autoregression_params_->value());
    int dim = ans.nrow();
    if (weights.size() != dim) {
      report_error("Wrong size weight vector.");
    }
    ans *= weights[0];
    ConstVectorView shifted_weights(weights, 1);
    VectorView(ans.diag(), 0, dim - 1) += shifted_weights;
    return ans;
  }
  
  void AutoRegressionTransitionMatrix::add_to_block(SubMatrix block) const {
    check_can_add(block);
    block.row(0) += autoregression_params_->value();
    VectorView d(block.subdiag(1));
    d += 1;
  }

  Matrix AutoRegressionTransitionMatrix::dense() const {
    int p = nrow();
    Matrix ans(p, p, 0.0);
    ans.row(0) = autoregression_params_->value();
    ans.subdiag(1) = 1.0;
    return ans;
  }
  //======================================================================
  namespace {
    typedef SingleElementInFirstRow SEIFR;
  }  // namespace

  void SEIFR::multiply(VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    lhs = 0;
    lhs[0] = rhs[position_] * value_;
  }

  void SEIFR::multiply_and_add(VectorView lhs,
                               const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    lhs[0] += rhs[position_] * value_;
  }

  void SEIFR::Tmult(VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_cols(lhs.size());
    conforms_to_rows(rhs.size());
    lhs = 0;
    lhs[position_] = value_ * rhs[0];
  }

  void SEIFR::multiply_inplace(VectorView x) const {
    conforms_to_cols(x.size());
    double tmp = x[position_] * value_;
    x = 0;
    x[0] = tmp;
  }

  void SEIFR::matrix_multiply_inplace(SubMatrix m) const {
    conforms_to_cols(m.nrow());
    m.row(0) = value_ * m.row(position_);
    if (m.nrow() > 1) {
      SubMatrix(m, 1, m.nrow() - 1, 0, m.ncol() - 1) = 0;
    }
  }

  void SEIFR::matrix_transpose_premultiply_inplace(SubMatrix m) const {
    conforms_to_rows(m.nrow());
    m.col(0) = m.col(position_) * value_;
    SubMatrix(m, 0, m.nrow() - 1, 1, m.ncol() - 1) = 0;
  }

  SpdMatrix SEIFR::inner() const {
    SpdMatrix ans(ncol(), 0.0);
    ans(position_, position_) = square(value_);
    return ans;
  }

  SpdMatrix SEIFR::inner(const ConstVectorView &weights) const {
    SpdMatrix ans(ncol(), 0.0);
    ans(position_, position_) = square(value_) * weights[0];
    return ans;
  }
  
  void SEIFR::add_to_block(SubMatrix block) const {
    conforms_to_rows(block.nrow());
    conforms_to_cols(block.ncol());
    block(0, position_) += value_;
  }

  //======================================================================
  GenericSparseMatrixBlockElementProxy &GenericSparseMatrixBlockElementProxy::
  operator=(double new_value) {
    matrix_->insert_element(row_, col_, new_value);
    value_ = new_value;
    return *this;
  }

  GenericSparseMatrixBlock::GenericSparseMatrixBlock(int nrow, int ncol)
      : nrow_(nrow),
        ncol_(ncol),
        nrow_compressed_(0),
        empty_row_(ncol_),
        empty_column_(nrow_) {
    if (nrow < 0 || ncol < 0) {
      report_error("Negative matrix dimension.");
    }
  }

  GenericSparseMatrixBlockElementProxy GenericSparseMatrixBlock::operator()(
      int row, int col) {
    auto it = rows_.find(row);
    if (it == rows_.end()) {
      return GenericSparseMatrixBlockElementProxy(row, col, 0, this);
    } else {
      return GenericSparseMatrixBlockElementProxy(row, col, it->second[col],
                                                  this);
    }
  }

  double GenericSparseMatrixBlock::operator()(int row, int col) const {
    auto it = rows_.find(row);
    if (it == rows_.end()) {
      return 0;
    } else {
      return it->second[col];
    }
  }

  void GenericSparseMatrixBlock::set_row(const SparseVector &row,
                                         int row_number) {
    if (row.size() != ncol()) {
      report_error("Size of inserted row must match the number of columns.");
    }
    auto it = rows_.find(row_number);
    if (it == rows_.end()) {
      ++nrow_compressed_;
    }
    rows_[row_number] = row;
    for (const auto &el : row) {
      insert_element_in_columns(row_number, el.first, el.second);
    }
  }

  void GenericSparseMatrixBlock::set_column(const SparseVector &column,
                                            int col_number) {
    if (column.size() != nrow()) {
      report_error("Size of inserted column must match the number of rows.");
    }
    columns_[col_number] = column;
    for (const auto &el : column) {
      insert_element_in_rows(el.first, col_number, el.second);
    }
  }

  void GenericSparseMatrixBlock::insert_element_in_columns(uint row, uint col,
                                                           double value) {
    auto it = columns_.find(col);
    if (it == columns_.end()) {
      SparseVector column_vector(nrow_);
      column_vector[row] = value;
      columns_.insert(std::make_pair(col, column_vector));
    } else {
      it->second[row] = value;
    }
  }

  void GenericSparseMatrixBlock::insert_element_in_rows(uint row, uint col,
                                                        double value) {
    auto it = rows_.find(row);
    if (it == rows_.end()) {
      SparseVector row_vector(ncol_);
      row_vector[col] = value;
      rows_.insert(std::make_pair(row, row_vector));
      ++nrow_compressed_;
    } else {
      it->second[col] = value;
    }
  }

  void GenericSparseMatrixBlock::multiply(VectorView lhs,
                                          const ConstVectorView &rhs) const {
    lhs = 0.0;
    multiply_and_add(lhs, rhs);
  }

  void GenericSparseMatrixBlock::multiply_and_add(
      VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_cols(rhs.size());
    conforms_to_rows(lhs.size());
    for (const auto &row : rows_) {
      lhs[row.first] += row.second.dot(rhs);
    }
  }

  void GenericSparseMatrixBlock::Tmult(VectorView lhs,
                                       const ConstVectorView &rhs) const {
    conforms_to_rows(rhs.size());
    conforms_to_cols(lhs.size());
    lhs = 0;
    for (const auto &col : columns_) {
      lhs[col.first] = col.second.dot(rhs);
    }
  }

  void GenericSparseMatrixBlock::multiply_inplace(VectorView x) const {
    if (nrow() != ncol()) {
      report_error("multiply_inplace is only defined for square matrices.");
    }
    conforms_to_cols(x.size());
    Vector ans(nrow_compressed_);
    int counter = 0;
    for (const auto &row : rows_) {
      ans[counter++] = row.second.dot(x);
    }
    x = 0;
    counter = 0;
    for (const auto &row : rows_) {
      x[row.first] = ans[counter++];
    }
  }

  SpdMatrix GenericSparseMatrixBlock::inner() const {
    SpdMatrix ans(ncol(), 0.0);
    for (const auto &el : rows_) {
      el.second.add_outer_product(ans);
    }
    return ans;
  }

  SpdMatrix GenericSparseMatrixBlock::inner(
      const ConstVectorView &weights) const {
    SpdMatrix ans(ncol(), 0.0);
    for (const auto &el : rows_) {
      uint position = el.first;
      el.second.add_outer_product(ans, weights[position]);
    }
    return ans;
  }
  
  void GenericSparseMatrixBlock::add_to_block(SubMatrix block) const {
    conforms_to_rows(block.nrow());
    conforms_to_cols(block.ncol());
    for (const auto &row : rows_) {
      row.second.add_this_to(block.row(row.first), 1.0);
    }
  }

  const SparseVector &GenericSparseMatrixBlock::row(int row_number) const {
    const auto it = rows_.find(row_number);
    if (it == rows_.end()) {
      return empty_row_;
    } else {
      return it->second;
    }
  }

  const SparseVector &GenericSparseMatrixBlock::column(int col_number) const {
    const auto it = columns_.find(col_number);
    if (it == columns_.end()) {
      return empty_column_;
    } else {
      return it->second;
    }
  }

  //======================================================================
  void StackedRegressionCoefficients::add_row(const Ptr<GlmCoefs> &beta) {
    if (!coefficients_.empty()) {
      if (beta->nvars_possible() != coefficients_[0]->nvars_possible()) {
        report_error("All coefficient vectors must be the same size.");
      }
    }
    coefficients_.push_back(beta);
  }

  namespace {
    template <class VECTOR>
    Vector stacked_regression_vector_mult(
        const VECTOR &v, const StackedRegressionCoefficients &coef) {
      Vector ans(coef.nrow());
      for (int i = 0; i < coef.nrow(); ++i) {
        ans[i] = coef.coefficients(i).predict(v);
      }
      return ans;
    }
  }  // namespace
  
  Vector StackedRegressionCoefficients::operator*(
      const Vector &v) const {
    return stacked_regression_vector_mult(v, *this);
  }
  Vector StackedRegressionCoefficients::operator*(
      const VectorView &v) const {
    return stacked_regression_vector_mult(v, *this);
  }
  Vector StackedRegressionCoefficients::operator*(
      const ConstVectorView &v) const {
    return stacked_regression_vector_mult(v, *this);
  }

  Vector StackedRegressionCoefficients::Tmult(
      const ConstVectorView &x) const {
    Vector ans(ncol());
    for (int i = 0; i < ncol(); ++i) {
      ans[i] = 0;
      for (int j = 0; j < nrow(); ++j) {
        ans[i] += coefficients_[j]->value()[i] * x[i];
      }
    }
    return ans;
  }

  SpdMatrix StackedRegressionCoefficients::inner() const {
    SpdMatrix ans(ncol(), 0.0);
    for (int i = 0; i < nrow(); ++i) {
      ans.add_outer(coefficients_[i]->value(), coefficients_[i]->inc());
    }
    return ans;
  }

  SpdMatrix StackedRegressionCoefficients::inner(
      const ConstVectorView &weights) const {
    SpdMatrix ans(ncol(), 0.0);
    for (int i = 0; i < nrow(); ++i) {
      ans.add_outer(coefficients_[i]->value(),
                    coefficients_[i]->inc(), weights[i]);
    }
    return ans;
  }

  Matrix &StackedRegressionCoefficients::add_to(Matrix &P) const {
    for (int i = 0; i < nrow(); ++i) {
      P.row(i) += coefficients_[i]->value();
    }
    return P;
  }

  SubMatrix StackedRegressionCoefficients::add_to_submatrix(SubMatrix P) const {
    for (int i = 0; i < nrow(); ++i) {
      P.row(i) += coefficients_[i]->value();
    }
    return P;
  }
  
  //======================================================================
  Matrix SparseKalmanMatrix::operator*(const Matrix &rhs) const {
    int nr = nrow();
    int nc = rhs.ncol();
    Matrix ans(nr, nc);
    for (int i = 0; i < nc; ++i) {
      ans.col(i) = (*this) * rhs.col(i);
    }
    return ans;
  }

  Matrix SparseKalmanMatrix::Tmult(const Matrix &rhs) const {
    Matrix ans(ncol(), rhs.ncol());
    for (int i = 0; i < rhs.ncol(); ++i) {
      ans.col(i) = this->Tmult(rhs.col(i));
    }
    return ans;
  }

  void SparseKalmanMatrix::sandwich_inplace(SpdMatrix &P) const {
    // First replace P with *this * P, which corresponds to *this
    // multiplying each column of P.
    for (int i = 0; i < P.ncol(); ++i) {
      P.col(i) = (*this) * P.col(i);
    }
    // Next, post-multiply P by this->transpose.  A * B is the same
    // thing as taking each row of A and transpose-multiplying it by
    // B.  (This follows because A * B = (B^T * A^T)^T ).  Because the
    // final factor of the product is this->transpose(), the
    // 'transpose-multiply' operation is really just a regular
    // multiplication.
    for (int i = 0; i < P.nrow(); ++i) {
      P.row(i) = (*this) * P.row(i);
    }
  }

  void SparseKalmanMatrix::sandwich_inplace_submatrix(SubMatrix P) const {
    SpdMatrix tmp(P.to_matrix());
    sandwich_inplace(tmp);
    P = tmp;
  }

  // Replaces P with this.transpose * P * this
  void SparseKalmanMatrix::sandwich_inplace_transpose(SpdMatrix &P) const {
    // First replace P with this->Tmult(P), which just
    // transpose-multiplies each column of P by *this.
    for (int i = 0; i < P.ncol(); ++i) {
      P.col(i) = this->Tmult(P.col(i));
    }
    // Next take the resulting matrix and post-multiply it by 'this',
    // which is just the transpose of this->transpose * that.
    for (int j = 0; j < P.nrow(); ++j) {
      P.row(j) = this->Tmult(P.row(j));
    }
  }

  // The logic of this function parallels "sandwich_inplace" above.
  // The implementation is different because for non-square matrices
  // we need a temporary variable to store intermediate results.
  SpdMatrix SparseKalmanMatrix::sandwich(const SpdMatrix &P) const {
    SpdMatrix ans(nrow());
    Matrix tmp(nrow(), ncol());
    for (int i = 0; i < ncol(); ++i) {
      tmp.col(i) = (*this) * P.col(i);
    }
    for (int i = 0; i < nrow(); ++i) {
      ans.row(i) = (*this) * tmp.row(i);
    }
    ans.fix_near_symmetry();
    return ans;
  }

  SpdMatrix SparseKalmanMatrix::sandwich_transpose(const SpdMatrix &P) const {
    SpdMatrix ans(ncol());
    Matrix tmp(ncol(), nrow());
    for (int i = 0; i < nrow(); ++i) {
      tmp.col(i) = this->Tmult(P.col(i));
    }
    for (int i = 0; i < ncol(); ++i) {
      ans.row(i) = this->Tmult(tmp.row(i));
    }
    return ans;
  }

  SubMatrix SparseKalmanMatrix::add_to_submatrix(SubMatrix P) const {
    Matrix tmp(P.to_matrix());
    this->add_to(tmp);
    P = tmp;
    return P;
  }

  // Returns this * rhs.transpose().
  Matrix SparseKalmanMatrix::multT(const Matrix &rhs) const {
    if (ncol() != rhs.ncol()) {
      report_error(
          "SparseKalmanMatrix::multT called with "
          "incompatible matrices.");
    }
    Matrix ans(nrow(), rhs.nrow());
    for (int i = 0; i < rhs.nrow(); ++i) {
      ans.col(i) = (*this) * rhs.row(i);
    }
    return ans;
  }

  Matrix operator*(const Matrix &lhs, const SparseKalmanMatrix &rhs) {
    int nr = lhs.nrow();
    int nc = rhs.ncol();
    Matrix ans(nr, nc);
    for (int i = 0; i < nr; ++i) {
      ans.row(i) = rhs.Tmult(lhs.row(i));
    }
    return ans;
  }

  // Returns lhs * rhs.transpose().  This is the same as the transpose of
  // rhs.Tmult(lhs.transpose()), but of course lhs is symmetric.  The answer can
  // be computed by filling the rows of the solution with
  // rhs.Tmult(columns_of_lhs).
  Matrix multT(const SpdMatrix &lhs, const SparseKalmanMatrix &rhs) {
    Matrix ans(lhs.nrow(), rhs.nrow());
    for (int i = 0; i < ans.nrow(); ++i) {
      ans.row(i) = rhs * lhs.col(i);
    }
    return ans;
  }

  //======================================================================
  BlockDiagonalMatrix::BlockDiagonalMatrix() : nrow_(0), ncol_(0) {}

  void BlockDiagonalMatrix::add_block(const Ptr<SparseMatrixBlock> &m) {
    blocks_.push_back(m);
    nrow_ += m->nrow();
    ncol_ += m->ncol();
    row_boundaries_.push_back(nrow_);
    col_boundaries_.push_back(ncol_);
  }

  void BlockDiagonalMatrix::replace_block(int which_block,
                                          const Ptr<SparseMatrixBlock> &b) {
    if (b->nrow() != blocks_[which_block]->nrow() ||
        b->ncol() != blocks_[which_block]->ncol()) {
      report_error("");
    }
    blocks_[which_block] = b;
  }

  void BlockDiagonalMatrix::clear() {
    blocks_.clear();
    nrow_ = ncol_ = 0;
    row_boundaries_.clear();
    col_boundaries_.clear();
  }

  int BlockDiagonalMatrix::nrow() const { return nrow_; }
  int BlockDiagonalMatrix::ncol() const { return ncol_; }

  // TODO(user): add a unit test for the case where diagonal
  // blocks are not square.
  Vector block_multiply(const ConstVectorView &v, int nrow, int ncol,
                        const std::vector<Ptr<SparseMatrixBlock>> &blocks_) {
    if (v.size() != ncol) {
      report_error(
          "incompatible vector in "
          "BlockDiagonalMatrix::operator*");
    }
    Vector ans(nrow);

    int lhs_pos = 0;
    int rhs_pos = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int nr = blocks_[b]->nrow();
      VectorView lhs(ans, lhs_pos, nr);
      lhs_pos += nr;

      int nc = blocks_[b]->ncol();
      ConstVectorView rhs(v, rhs_pos, nc);
      rhs_pos += nc;
      blocks_[b]->multiply(lhs, rhs);
    }
    return ans;
  }

  Vector BlockDiagonalMatrix::operator*(const Vector &v) const {
    return block_multiply(ConstVectorView(v), nrow(), ncol(), blocks_);
  }

  Vector BlockDiagonalMatrix::operator*(const VectorView &v) const {
    return block_multiply(ConstVectorView(v), nrow(), ncol(), blocks_);
  }
  Vector BlockDiagonalMatrix::operator*(const ConstVectorView &v) const {
    return block_multiply(v, nrow(), ncol(), blocks_);
  }

  Vector BlockDiagonalMatrix::Tmult(const ConstVectorView &x) const {
    if (x.size() != nrow()) {
      report_error(
          "incompatible vector in "
          "BlockDiagonalMatrix::Tmult");
    }
    int lhs_pos = 0;
    int rhs_pos = 0;
    Vector ans(ncol(), 0);

    for (int b = 0; b < blocks_.size(); ++b) {
      VectorView lhs(ans, lhs_pos, blocks_[b]->ncol());
      lhs_pos += blocks_[b]->ncol();
      ConstVectorView rhs(x, rhs_pos, blocks_[b]->nrow());
      rhs_pos += blocks_[b]->nrow();
      blocks_[b]->Tmult(lhs, rhs);
    }
    return ans;
  }

  SpdMatrix BlockDiagonalMatrix::inner() const {
    SpdMatrix ans(ncol(), 0.0);
    int start = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int end = start + blocks_[b]->ncol();
      SubMatrix(ans, start, end - 1, start, end - 1)
          = blocks_[b]->inner();
      start = end;
    }
    return ans;
  }

  SpdMatrix BlockDiagonalMatrix::inner(const ConstVectorView &weights) const {
    if (weights.size() != nrow()) {
      report_error("Wrong size weight vector for BlockDiagonalMatrix.");
    }
    SpdMatrix ans(ncol(), 0.0);
    int ans_start = 0;
    int weight_start = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      const SparseMatrixBlock &block(*blocks_[b]);
      int ans_end = ans_start + block.ncol();
      const ConstVectorView local_weights(weights, weight_start, block.nrow());
      SubMatrix(ans, ans_start, ans_end - 1, ans_start, ans_end - 1)
          = block.inner(local_weights);
      ans_start += block.ncol();
      weight_start += block.nrow();
    }
    return ans;
  }

  // this * P * this.transpose.
  SpdMatrix BlockDiagonalMatrix::sandwich(const SpdMatrix &P) const {
    // If *this is rectangular then the result will not be the same dimension as
    // P.  P must be ncol() X ncol().
    if (ncol() != P.nrow()) {
      report_error("'sandwich' called on a non-conforming matrix.");
    }
    SpdMatrix ans(nrow());
    Matrix workspace;
    for (int i = 0; i < blocks_.size(); ++i) {
      const SparseMatrixBlock &left(*(blocks_[i]));
      for (int j = i; j < blocks_.size(); ++j) {
        const SparseMatrixBlock &right(*(blocks_[j]));
        // The source matrix is determined by columns.  The number of columns in
        // the left block determines the number of rows in the source block.
        int rlo = (i == 0) ? 0 : col_boundaries_[i - 1];
        int rhi = col_boundaries_[i] - 1;
        // The number of rows in the transpose of the right block (i.e. the
        // number of columns in the right block) determines the number of
        // columns in the source block.
        int clo = (j == 0) ? 0 : col_boundaries_[j - 1];
        int chi = col_boundaries_[j] - 1;
        ConstSubMatrix source(P, rlo, rhi, clo, chi);

        // The destination block is determined by row boundaries.  The number of
        // rows in the left block determines the number of rows in the
        // destination block.
        rlo = (i == 0) ? 0 : row_boundaries_[i - 1];
        rhi = row_boundaries_[i] - 1;
        // The number of columns in the destination block is the number of
        // columns in the transpose of the right block (i.e. the number of rows
        // in the right block).
        clo = (j == 0) ? 0 : row_boundaries_[j - 1];
        chi = row_boundaries_[j] - 1;
        SubMatrix dest(ans, rlo, rhi, clo, chi);
        workspace.resize(left.nrow(), right.nrow());
        sandwich_block(left, right, source, dest, workspace);
      }
    }
    ans.reflect();
    return ans;
  }

  void BlockDiagonalMatrix::sandwich_inplace(SpdMatrix &P) const {
    for (int i = 0; i < blocks_.size(); ++i) {
      int rlo = i == 0 ? 0 : row_boundaries_[i - 1];
      int rhi = row_boundaries_[i] - 1;
      blocks_[i]->matrix_multiply_inplace(
          SubMatrix(P, rlo, rhi, 0, P.ncol() - 1));
    }
    for (int i = 0; i < blocks_.size(); ++i) {
      int clo = i == 0 ? 0 : col_boundaries_[i - 1];
      int chi = col_boundaries_[i] - 1;
      blocks_[i]->matrix_transpose_premultiply_inplace(
          SubMatrix(P, 0, P.nrow() - 1, clo, chi));
    }
  }

  void BlockDiagonalMatrix::sandwich_inplace_submatrix(SubMatrix P) const {
    for (int i = 0; i < blocks_.size(); ++i) {
      for (int j = 0; j < blocks_.size(); ++j) {
        sandwich_inplace_block((*blocks_[i]), (*blocks_[j]),
                               get_submatrix_block(P, i, j));
      }
    }
  }

  void BlockDiagonalMatrix::sandwich_inplace_block(
      const SparseMatrixBlock &left, const SparseMatrixBlock &right,
      SubMatrix middle) const {
    for (int i = 0; i < middle.ncol(); ++i) {
      left.multiply_inplace(middle.col(i));
    }

    for (int i = 0; i < middle.nrow(); ++i) {
      right.multiply_inplace(middle.row(i));
    }
  }

  // Fills dest with left * source * right.transpose.
  void BlockDiagonalMatrix::sandwich_block(const SparseMatrixBlock &left,
                                           const SparseMatrixBlock &right,
                                           const ConstSubMatrix &source,
                                           SubMatrix &dest,
                                           Matrix &workspace) const {
    // Workspace will hold the reult of left * source.
    workspace.resize(left.nrow(), source.ncol());
    for (int i = 0; i < source.ncol(); ++i) {
      left.multiply(workspace.col(i), source.col(i));
    }
    // Now put the result of workspace * right^T into dest.  We can do this by
    // putting the result of right * workspace^T into dest^T.
    for (int i = 0; i < workspace.nrow(); ++i) {
      // We want workspace * right^T.  The transpose of this is right *
      // workspace^T.  Multiply right by each row of workspace, and place the
      // result in the rows of dest.
      right.multiply(dest.row(i), workspace.row(i));
    }
  }

  SubMatrix BlockDiagonalMatrix::get_block(Matrix &m, int i, int j) const {
    int rlo = (i == 0 ? 0 : row_boundaries_[i - 1]);
    int rhi = row_boundaries_[i] - 1;

    int clo = (j == 0 ? 0 : col_boundaries_[j - 1]);
    int chi = col_boundaries_[j] - 1;
    return SubMatrix(m, rlo, rhi, clo, chi);
  }

  SubMatrix BlockDiagonalMatrix::get_submatrix_block(SubMatrix m, int i,
                                                     int j) const {
    int rlo = (i == 0 ? 0 : row_boundaries_[i - 1]);
    int rhi = row_boundaries_[i] - 1;

    int clo = (j == 0 ? 0 : col_boundaries_[j - 1]);
    int chi = col_boundaries_[j] - 1;
    return SubMatrix(m, rlo, rhi, clo, chi);
  }

  Matrix &BlockDiagonalMatrix::add_to(Matrix &P) const {
    for (int b = 0; b < blocks_.size(); ++b) {
      SubMatrix block = get_block(P, b, b);
      blocks_[b]->add_to_block(block);
    }
    return P;
  }

  SubMatrix BlockDiagonalMatrix::add_to_submatrix(SubMatrix P) const {
    for (int b = 0; b < blocks_.size(); ++b) {
      SubMatrix block = get_submatrix_block(P, b, b);
      blocks_[b]->add_to_block(block);
    }
    return P;
  }

  Vector BlockDiagonalMatrix::left_inverse(const ConstVectorView &rhs) const {
    if (rhs.size() != nrow()) {
      report_error("Wrong size argument passed to left_inverse().");
    }
    Vector ans(ncol());
    int lhs_pos = 0;
    int rhs_pos = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      ConstVectorView rhs_block(rhs, rhs_pos, blocks_[b]->nrow());
      VectorView lhs(ans, lhs_pos, blocks_[b]->ncol());
      blocks_[b]->left_inverse(lhs, rhs_block);
    }
    return ans;
  }
  
  //===========================================================================
  namespace {
    template <class VECTOR>
    Vector block_multiply_impl(
        const std::vector<Ptr<SparseMatrixBlock>> &blocks, const VECTOR &rhs) {
      Vector ans(blocks.back()->nrow(), 0.0);
      int start = 0;
      for (int i = 0; i < blocks.size(); ++i) {
        int ncol = blocks[i]->ncol();
        blocks[i]->multiply_and_add(VectorView(ans),
                                    ConstVectorView(rhs, start, ncol));
        start += ncol;
      }
      return ans;
    }
  }  // namespace

  void SparseVerticalStripMatrix::add_block(
      const Ptr<SparseMatrixBlock> &block) {
    if (!blocks_.empty() && block->nrow() != blocks_.back()->nrow()) {
      report_error("All blocks must have the same number of rows");
    }
    blocks_.push_back(block);
    ncol_ += block->ncol();
  }

  Vector SparseVerticalStripMatrix::operator*(const Vector &v) const {
    check_can_multiply(v.size());
    return block_multiply_impl(blocks_, v);
  }
  Vector SparseVerticalStripMatrix::operator*(const VectorView &v) const {
    check_can_multiply(v.size());
    return block_multiply_impl(blocks_, v);
  }
  Vector SparseVerticalStripMatrix::operator*(const ConstVectorView &v) const {
    check_can_multiply(v.size());
    return block_multiply_impl(blocks_, v);
  }

  Vector SparseVerticalStripMatrix::Tmult(const ConstVectorView &v) const {
    check_can_Tmult(v.size());
    Vector ans(ncol());
    int start = 0;
    for (int i = 0; i < blocks_.size(); ++i) {
      int dim = blocks_[i]->ncol();
      blocks_[i]->Tmult(VectorView(ans, start, dim), v);
      start += dim;
    }
    return ans;
  }

  SpdMatrix SparseVerticalStripMatrix::inner() const {
    SpdMatrix ans(ncol(), 0.0);
    std::vector<Matrix> dense_blocks;
    dense_blocks.reserve(blocks_.size());
    for (int b = 0; b < blocks_.size(); ++b) {
      dense_blocks.push_back(blocks_[b]->dense());
    }
    int row_start = 0;
    for (int b0 = 0; b0 < blocks_.size(); ++b0) {
      BlockDiagonalMatrix row_block;
      row_block.add_block(blocks_[b0]);
      int col_start = row_start;
      int row_end = row_start + blocks_[b0]->ncol();
      for (int b1 = b0; b1 < blocks_.size(); ++b1) {
        int col_end = col_start + blocks_[b1]->ncol();
        SubMatrix(ans, row_start, row_end - 1, col_start, col_end - 1)
            = row_block.Tmult(dense_blocks[b1]);
        col_start = col_end;
      }
      row_start = row_end;
    }
    ans.reflect();
    return ans;
  }

  SpdMatrix SparseVerticalStripMatrix::inner(
      const ConstVectorView &weights) const {
    SpdMatrix ans(ncol(), 0.0);
    std::vector<Matrix> dense_blocks;
    dense_blocks.reserve(blocks_.size());
    DiagonalMatrix weight_block(weights);
    for (int b = 0; b < blocks_.size(); ++b) {
      dense_blocks.push_back(weight_block * blocks_[b]->dense());
    }
    
    int row_start = 0;
    for (int b0 = 0; b0 < blocks_.size(); ++b0) {
      BlockDiagonalMatrix row_block;
      row_block.add_block(blocks_[b0]);
      int col_start = row_start;
      int row_end = row_start + blocks_[b0]->ncol();
      for (int b1 = b0; b1 < blocks_.size(); ++b1) {
        int col_end = col_start + blocks_[b1]->ncol();
        SubMatrix(ans, row_start, row_end - 1, col_start, col_end - 1)
            = row_block.Tmult(dense_blocks[b1]);
        col_start = col_end;
      }
      row_start = row_end;
    }
    ans.reflect();
    return ans;
  }
  
  Matrix &SparseVerticalStripMatrix::add_to(Matrix &P) const {
    check_can_add(P.nrow(), P.ncol());
    int start_column = 0;
    for (int i = 0; i < blocks_.size(); ++i) {
      int ncol = blocks_[i]->ncol();
      blocks_[i]->add_to_block(
          SubMatrix(P, 0, nrow() - 1, start_column, start_column + ncol - 1));
      start_column += ncol;
    }
    return P;
  }

  SubMatrix SparseVerticalStripMatrix::add_to_submatrix(SubMatrix P) const {
    check_can_add(P.nrow(), P.ncol());
    int start_column = 0;
    for (int i = 0; i < blocks_.size(); ++i) {
      int ncol = blocks_[i]->ncol();
      blocks_[i]->add_to_block(
          SubMatrix(P, 0, nrow() - 1, start_column, start_column + ncol - 1));
      start_column += ncol;
    }
    return P;
  }

  void SparseVerticalStripMatrix::check_can_multiply(int vector_size) const {
    if (ncol() != vector_size) {
      report_error("Incompatible vector multiplication.");
    }
  }
  void SparseVerticalStripMatrix::check_can_Tmult(int vector_size) const {
    if (nrow() != vector_size) {
      std::ostringstream err;
      err << "Incompatible vector (transpose-)multiplication.  "
          << "This matrix has " << nrow() << " rows.  The target vector has "
          << vector_size << " elements." << std::endl;
      report_error(err.str());
    }
  }

  void SparseVerticalStripMatrix::check_can_add(int rows, int cols) const {
    if (nrow() != rows || ncol() != cols) {
      report_error("Incompatible matrix addition.");
    }
  }
}  // namespace BOOM
