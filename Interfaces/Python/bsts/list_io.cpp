// Copyright 2011 Google Inc. All Rights Reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

#include <string>
#include "list_io.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  void PythonListIoManager::add_list_element(PythonListIoElement *element) {
    elements_.push_back(std::shared_ptr<PythonListIoElement>(element));
  }

  void PythonListIoManager::prepare_to_write(int niter) {
    for (int i = 0; i < elements_.size(); ++i) {
      elements_[i]->prepare_to_write(niter);
    }
  }

  void PythonListIoManager::prepare_to_stream() {
    for (int i = 0; i < elements_.size(); ++i) {
      elements_[i]->prepare_to_stream();
    }    
  }

  void PythonListIoManager::write() {
    for (int i = 0; i < elements_.size(); ++i) {
      elements_[i]->write();
    }
  }

  void PythonListIoManager::stream() {
    for (int i = 0; i < elements_.size(); ++i) {
      elements_[i]->stream();
    }
  }

  std::string PythonListIoManager::repr() const {
    std::ostringstream msg;
    msg << "PythonListIoManager:number of elements: " << elements_.size() << endl;
    for (int i = 0; i < elements_.size(); ++i) {
      msg << elements_[i]->repr();
    }
    return msg.str();
  }

  void PythonListIoManager::advance(int n) {
    for (int i = 0; i < elements_.size(); ++i) {
      elements_[i]->advance(n);
    }
  }

  Vector PythonListIoManager::results(const std::string &name) const {
    Vector results = Vector();
    for (int i = 0; i < elements_.size(); ++i) {
      if (elements_[i]->name() == name) {
        results = elements_[i]->results();
        break;
      }
    }
    return results;
  }

  //======================================================================
  PythonListIoElement::PythonListIoElement(const std::string &name) : name_(name), position_(0) {}

  PythonListIoElement::~PythonListIoElement() {}

  void PythonListIoElement::prepare_to_stream() {
    position_ = 0;
  }

  int PythonListIoElement::next_position() {
    return position_++;
  }

  const std::string &PythonListIoElement::name() const { return name_; }

  std::string PythonListIoElement::repr() const {
    std::ostringstream msg;
    msg << "PythonListIoElement:name: " << name() << ", position: " << position_ << endl;
    return msg.str();
  }

  void PythonListIoElement::advance(int n) {position_ += n;}

  //======================================================================
  RealValuedPythonListIoElement::RealValuedPythonListIoElement(const std::string &name)
      : PythonListIoElement(name)
  {}

  double * RealValuedPythonListIoElement::data() { return internal_data_.data(); }

  std::string RealValuedPythonListIoElement::repr() const {
    std::ostringstream msg;
    msg << name() << ":internal_data.size:" << internal_data_.size() << endl;
    for (int i=0; i < internal_data_.size(); i++) {
      msg << name() << ":element[" << i << "]: " << internal_data_[i] << endl; 
    }
    msg << endl;
    return PythonListIoElement::repr() + msg.str();
  }

  void RealValuedPythonListIoElement::resize(unsigned int size) {
    internal_data_.resize(size);
  }

  void RealValuedPythonListIoElement::prepare_to_write(int niter) {
    resize(niter);
  }

  Vector RealValuedPythonListIoElement::results() const {
    return internal_data_;
  }

  //======================================================================
  PartialSpdListElement::PartialSpdListElement(const Ptr<SpdParams> &prm,
                                               const std::string &name,
                                               int which, bool
                                               report_sd)
      : RealValuedPythonListIoElement(name),
        prm_(prm),
        which_(which),
        report_sd_(report_sd) {}

  void PartialSpdListElement::write() {
    CheckSize();
    double variance = prm_->var()(which_, which_);
    data()[next_position()] = report_sd_ ? sqrt(variance) : variance;
  }

  void PartialSpdListElement::stream() {
    CheckSize();
    SpdMatrix Sigma = prm_->var();
    double v = data()[next_position()];
    if (report_sd_) v *= v;
    Sigma(which_, which_) = v;
    prm_->set_var(Sigma);
  }

  void PartialSpdListElement::CheckSize() {
    if (nrow(prm_->var()) <= which_) {
      std::ostringstream err;
      err << "Sizes do not match in PartialSpdListElement..."
          << endl
          << "Matrix has " << nrow(prm_->var()) << " rows, but "
          << "you're trying to access row " << which_
          << endl;
      report_error(err.str().c_str());
    }
  }
  //======================================================================
  UnivariateListElement::UnivariateListElement(const Ptr<UnivParams> &prm,
                                               const std::string &name)
      : RealValuedPythonListIoElement(name),
        prm_(prm)
  {}

  void UnivariateListElement::write() {
    data()[next_position()] = prm_->value();
  }

  void UnivariateListElement::stream() {
    prm_->set(data()[next_position()]);
  }

  //======================================================================
  StandardDeviationListElement::StandardDeviationListElement(
      const Ptr<UnivParams> &variance, const std::string &name)
      : RealValuedPythonListIoElement(name),
        variance_(variance)
  {}

  void StandardDeviationListElement::write() {
    data()[next_position()] = sqrt(variance_->value());
  }

  void StandardDeviationListElement::stream() {
    double sd = data()[next_position()];
    variance_->set(square(sd));
  }

  //======================================================================
  NativeUnivariateListElement::NativeUnivariateListElement(
      ScalarIoCallback *callback,
      const std::string &name,
      double *streaming_buffer)
      : RealValuedPythonListIoElement(name),
        streaming_buffer_(streaming_buffer),
        vector_view_(0, 0, 0)
  {
    if (callback) {
      callback_.reset(callback);
    }
  }

  void NativeUnivariateListElement::prepare_to_write(int niter) {
    if (!callback_) {
      report_error(
          "NULL callback in NativeUnivariateListElement::prepare_to_write");
    }
    resize(niter);
    vector_view_.reset(data(), niter, 1);
  }

  void NativeUnivariateListElement::write() {
    vector_view_[next_position()] = callback_->get_value();
  }

  void NativeUnivariateListElement::stream() {
    if(streaming_buffer_){
      *streaming_buffer_ = vector_view_[next_position()];
    }
  }

  //======================================================================
  VectorListElement::VectorListElement(
      const Ptr<VectorParams> &prm,
      const std::string &name,
      const std::vector<std::string> &element_names)
      : RealValuedPythonListIoElement(name),
        prm_(prm),
        matrix_view_(0, 0, 0),
        element_names_(element_names)
  {}

  void VectorListElement::prepare_to_write(int niter) {
    // The call to size should not return the minimial size.  It
    // should return the full size, because that's what we're going to
    // write to the R list.
    int dim = prm_->size(false);
    resize(niter * dim);
    matrix_view_.reset(SubMatrix(data(), niter, dim));
  }

  void VectorListElement::write() {
    CheckSize();
    matrix_view_.row(next_position()) = prm_->value();
  }

  void VectorListElement::stream() {
    CheckSize();
    prm_->set(matrix_view_.row(next_position()));
  }

  void VectorListElement::CheckSize() {
    if (matrix_view_.ncol() != prm_->size(false)) {
      std::ostringstream err;
      err << "sizes do not match in VectorListElement::stream/write..."
          << endl
          << "buffer has space for " << matrix_view_.ncol() << " elements, "
          << " but you're trying to access " << prm_->size(false)
          ;
      report_error(err.str().c_str());
    }
  }
  //======================================================================
  GlmCoefsListElement::GlmCoefsListElement(
      const Ptr<GlmCoefs> &coefs,
      const std::string &param_name,
      const std::vector<std::string> &element_names)
      : VectorListElement(coefs, param_name, element_names),
        coefs_(coefs)
  {}

  void GlmCoefsListElement::stream() {
    VectorListElement::stream();
    beta_ = coefs_->Beta();
    coefs_->set_Beta(beta_);
    coefs_->infer_sparsity();
  }

  //======================================================================
  SdVectorListElement::SdVectorListElement(const Ptr<VectorParams> &prm,
                                           const std::string &name)
      : RealValuedPythonListIoElement(name),
        prm_(prm),
        matrix_view_(0, 0, 0)
  {}

  void SdVectorListElement::prepare_to_write(int niter) {
    int dim = prm_->size();
    resize(niter * dim);
    matrix_view_.reset(SubMatrix(data(), niter, dim));
  }

  void SdVectorListElement::write() {
    CheckSize();
    matrix_view_.row(next_position()) = sqrt(prm_->value());
  }

  void SdVectorListElement::stream() {
    CheckSize();
    Vector sd = matrix_view_.row(next_position());
    prm_->set(sd * sd);
  }

  void SdVectorListElement::CheckSize() {
    if (matrix_view_.ncol() != prm_->size(false)) {
      std::ostringstream err;
      err << "sizes do not match in SdVectorListElement::stream/write..."
          << endl
          << "buffer has space for " << matrix_view_.ncol() << " elements, "
          << " but you're trying to access " << prm_->size(false)
          ;
      report_error(err.str().c_str());
    }
  }


  //======================================================================
  const std::vector<std::string> & MatrixListElementBase::row_names()const{
    return row_names_;
  }

  const std::vector<std::string> & MatrixListElementBase::col_names()const{
    return row_names_;
  }

  void MatrixListElementBase::set_row_names(
      const std::vector<std::string> &row_names){
    row_names_ = row_names;
  }

  void MatrixListElementBase::set_col_names(
      const std::vector<std::string> &col_names){
    col_names_ = col_names;
  }

  //======================================================================

  MatrixListElement::MatrixListElement(const Ptr<MatrixParams> &m,
                                       const std::string &param_name)
      : MatrixListElementBase(param_name),
        prm_(m),
        array_view_(0, Array::index3(0, 0, 0))
  {}

  void MatrixListElement::prepare_to_write(int niter) {
    int nr = prm_->nrow();
    int nc = prm_->ncol();
    resize(niter * nr * nc);
    array_view_.reset(data(), Array::index3(niter, nr, nc));
  }

  void MatrixListElement::write() {
    CheckSize();
    const Matrix &m(prm_->value());
    int iteration = next_position();
    int nr = m.nrow();
    int nc = m.ncol();
    for(int i = 0; i < nr; ++i){
      for(int j = 0; j < nc; ++j){
        array_view_(iteration, i, j) = m(i, j);
      }
    }
  }

  void MatrixListElement::stream() {
    CheckSize();
    int iteration = next_position();
    int nr = prm_->nrow();
    int nc = prm_->ncol();
    Matrix tmp(nr, nc);
    for(int i = 0; i < nr; ++i) {
      for(int j = 0; j < nc; ++j) {
      tmp(i, j) = array_view_(iteration, i, j);
      }
    }
    prm_->set(tmp);
  }

  int MatrixListElement::nrow()const {
    return prm_->nrow();
  }

  int MatrixListElement::ncol()const {
    return prm_->ncol();
  }

  void MatrixListElement::CheckSize() {
    const std::vector<int> & dims(array_view_.dim());
    const Matrix & value(prm_->value());
    if(value.nrow() != dims[1] ||
       value.ncol() != dims[2]) {
      std::ostringstream err;
      err << "sizes do not match in MatrixListElement::stream/write..."
          << endl
          << "dimensions of buffer:    [" << dims[0] << ", " << dims[1] << ", "
          << dims[2] << "]." <<endl
          << "dimensions of parameter: [" << value.nrow() << ", "
          << value.ncol() << "].";
      report_error(err.str().c_str());
    }
  }

  //======================================================================
  HierarchicalVectorListElement::HierarchicalVectorListElement(
      const std::vector<Ptr<VectorParams>> &parameters,
      const std::string &param_name)
      : RealValuedPythonListIoElement(param_name),
        array_view_(0, Array::index3(0, 0, 0))
  {
    parameters_.resize(parameters.size());
    for (int i = 0; i < parameters.size(); ++i) {
      add_vector(parameters[i]);
    }
  }

  HierarchicalVectorListElement::HierarchicalVectorListElement(
      const std::string &param_name)
      : RealValuedPythonListIoElement(param_name),
        array_view_(0, Array::index3(0, 0, 0))
  {}

  void HierarchicalVectorListElement::add_vector(const Ptr<VectorParams> &v) {
    if (!v) {
      report_error("Null pointer passed to HierarchicalVectorListElement");
    }
    if (!parameters_.empty()) {
      if (v->dim() != parameters_[0]->dim()) {
        report_error(
            "All parameters passed to HierarchicalVectorListElement "
            "must be the same size");
      }
    }
    parameters_.push_back(v);
  }

  void HierarchicalVectorListElement::prepare_to_write(int niter) {
    int number_of_groups = parameters_.size();
    int dim = parameters_[0]->dim();
    resize(niter * number_of_groups * dim);
    array_view_.reset(data(), Array::index3(niter, number_of_groups, dim));
  }

  void HierarchicalVectorListElement::write() {
    CheckSize();
    int iteration = next_position();
    int dimension = parameters_[0]->dim();
    for (int i = 0; i < parameters_.size(); ++i) {
      const Vector &value(parameters_[i]->value());
      for (int j = 0; j < dimension; ++j) {
        array_view_(iteration, i, j) = value[j];
      }
    }
  }

  void HierarchicalVectorListElement::stream() {
    CheckSize();
    int iteration = next_position();
    int dimension = parameters_[0]->dim();
    Vector values(dimension);
    for (int i = 0; i < parameters_.size(); ++i) {
      for (int j = 0; j < dimension; ++j) {
        values[j] = array_view_(iteration, i, j);
      }
      parameters_[i]->set(values);
    }
  }

  void HierarchicalVectorListElement::set_group_names(
      const std::vector<std::string> &group_names) {
    if (group_names.size() != parameters_.size()) {
      report_error("Vector of group names must be the same size as the "
                   "number of groups.");
    }
    group_names_ = group_names;
  }
  

  void HierarchicalVectorListElement::CheckSize() {
    const std::vector<int> &dims(array_view_.dim());
    if (dims[1] != parameters_.size() ||
        dims[2] != parameters_[0]->dim()) {
      std::ostringstream err;
      err << "sizes do not match in HierarchicalVectorListElement::"
          "stream/write..."
          << endl
          << "dimensions of buffer:    [" << dims[0] << ", " << dims[1] << ", "
          << dims[2] << "]." <<endl
          << "number of groups:    " << parameters_.size() << endl
          << "parameter dimension: " << parameters_[0]->dim() << "." << endl;
      report_error(err.str().c_str());
    }
  }
  //======================================================================
  UnivariateCollectionListElement::UnivariateCollectionListElement(
      const std::vector<Ptr<UnivParams>> &parameters,
      const std::string &name)
      : RealValuedPythonListIoElement(name),
        parameters_(parameters)
  {}

  void UnivariateCollectionListElement::prepare_to_write(int niter) {
    int dim = parameters_.size();
    resize(niter * dim);
    matrix_view_.reset(SubMatrix(data(), niter, dim));
  }

  void UnivariateCollectionListElement::write() {
    CheckSize();
    int row = next_position();
    for (int i = 0; i < parameters_.size(); ++i) {
      matrix_view_(row, i) = parameters_[i]->value();
    }
  }

  void UnivariateCollectionListElement::stream() {
    CheckSize();
    int row = next_position();
    for (int i = 0; i < parameters_.size(); ++i) {
      parameters_[i]->set(matrix_view_(row, i));
    }
  }

  void UnivariateCollectionListElement::CheckSize() {
    if (matrix_view_.ncol() != parameters_.size()) {
      std::ostringstream err;
      err << "The R buffer has " << matrix_view_.ncol()
          << " columns, but space is needed for "
          << parameters_.size() << " parameters.";
      report_error(err.str());
    }
  }

  //======================================================================
  void SdCollectionListElement::write() {
    CheckSize();
    int row = next_position();
    for (int i = 0; i < parameters().size(); ++i) {
      matrix_view()(row, i) = sqrt(parameters()[i]->value());
    }
  }

  void SdCollectionListElement::stream() {
    CheckSize();
    int row = next_position();
    for (int i = 0; i < parameters().size(); ++i) {
      parameters()[i]->set(square(matrix_view()(row, i)));
    }
  }

  //======================================================================

  SpdListElement::SpdListElement(const Ptr<SpdParams> &m,
                                 const std::string &param_name)
      : MatrixListElementBase(param_name),
        prm_(m),
        array_view_(0, Array::index3(0, 0, 0))
  {}

  void SpdListElement::prepare_to_write(int niter) {
    int dim = prm_->dim();
    resize(niter * dim * dim);
    array_view_.reset(data(), Array::index3(niter, dim, dim));
  }

  void SpdListElement::write() {
    CheckSize();
    const Matrix &m(prm_->value());
    int iteration = next_position();
    int nr = m.nrow();
    int nc = m.ncol();
    for(int i = 0; i < nr; ++i){
      for(int j = 0; j < nc; ++j){
        array_view_(iteration, i, j) = m(i, j);
      }
    }
  }

  void SpdListElement::stream() {
    CheckSize();
    int iteration = next_position();
    int nr = prm_->dim();
    int nc = prm_->dim();
    Matrix tmp(nr, nc);
    for(int i = 0; i < nr; ++i) {
      for(int j = 0; j < nc; ++j) {
      tmp(i, j) = array_view_(iteration, i, j);
      }
    }
    prm_->set(tmp);
  }

  int SpdListElement::nrow()const {
    return prm_->dim();
  }

  int SpdListElement::ncol()const {
    return prm_->dim();
  }

  void SpdListElement::CheckSize() {
    const std::vector<int> & dims(array_view_.dim());
    const Matrix & value(prm_->value());
    if(value.nrow() != dims[1] ||
       value.ncol() != dims[2]) {
      std::ostringstream err;
      err << "sizes do not match in SpdListElement::stream/write..."
          << endl
          << "dimensions of buffer:    [" << dims[0] << ", " << dims[1] << ", "
          << dims[2] << "]." <<endl
          << "dimensions of parameter: [" << value.nrow() << ", " << value.ncol()
          << "].";
      report_error(err.str().c_str());
    }
  }

  //======================================================================

  NativeVectorListElement::NativeVectorListElement(VectorIoCallback *callback,
                                                   const std::string &name,
                                                   Vector *vector_buffer)
      : RealValuedPythonListIoElement(name),
        streaming_buffer_(vector_buffer),
        matrix_view_(0, 0, 0),
        check_buffer_(true)
  {
    // Protect against a NULL callback.
    if (callback) {
      callback_.reset(callback);
    }
  }

  void NativeVectorListElement::prepare_to_write(int niter) {
    if (!callback_) {
      report_error(
          "NULL callback in NativeVectorListElement::prepare_to_write");
    }
    int dim = callback_->dim();
    resize(niter * dim);
    matrix_view_.reset(SubMatrix(data(), niter, dim));
    if (matrix_view_.ncol() != callback_->dim()) {
      report_error(
          "wrong size buffer set up for NativeVectorListElement::write");
    }
  }

  VectorView NativeVectorListElement::next_row() {
    return matrix_view_.row(next_position());
  }
  
  void NativeVectorListElement::write() {
    next_row() = callback_->get_vector();
  }

  void NativeVectorListElement::stream() {
    if (check_buffer_ && !streaming_buffer_) return;
    *streaming_buffer_ = next_row();
  }

  GenericVectorListElement::GenericVectorListElement(
      StreamableVectorIoCallback *callback,
      const std::string &name)
      : NativeVectorListElement(callback, name, nullptr)
  {
    if (callback) {
      callback_.reset(callback);
    } else {
      callback_.reset();
    }
  }

  
  //======================================================================
  NativeMatrixListElement::NativeMatrixListElement(MatrixIoCallback *callback,
                                                   const std::string &name,
                                                   Matrix *streaming_buffer)
      : MatrixListElementBase(name),
        streaming_buffer_(streaming_buffer),
        array_view_(0, ArrayBase::index3(0, 0, 0)),
        check_buffer_(true)
  {
    // Protect against NULL.
    if (callback) {
      callback_.reset(callback);
    }
  }

  void NativeMatrixListElement::prepare_to_write(int niter) {
    if (!callback_) {
      report_error(
          "NULL callback in NativeMatrixListElement::prepare_to_write.");
    }
    resize(niter * callback_->nrow() * callback_->ncol());
    array_view_.reset(data(),
                      Array::index3(niter,
                                    callback_->nrow(),
                                    callback_->ncol()));
  }

  void NativeMatrixListElement::write() {
    Matrix tmp = callback_->get_matrix();
    int niter = next_position();
    for (int i = 0; i < callback_->nrow(); ++i) {
      for (int j = 0; j < callback_->ncol(); ++j) {
        array_view_(niter, i, j) = tmp(i, j);
      }
    }
  }

  void NativeMatrixListElement::stream() {
    if (!streaming_buffer_) return;
    int niter = next_position();
    for (int i = 0; i < streaming_buffer_->nrow(); ++i) {
      for (int j = 0; j < streaming_buffer_->ncol(); ++j) {
        (*streaming_buffer_)(i, j) = array_view_(niter, i, j);
      }
    }
  }

  int NativeMatrixListElement::nrow()const {
    return callback_->nrow();
  }

  int NativeMatrixListElement::ncol()const {
    return callback_->ncol();
  }

  GenericMatrixListElement::GenericMatrixListElement(
      StreamableMatrixIoCallback *callback,
      const std::string &name)
      : NativeMatrixListElement(callback, name, nullptr)
  {
    if (callback) {
      callback_.reset(callback);
    } else {
      callback_.reset();
    }
  }

  void GenericMatrixListElement::stream() {
    if (!callback_) {
      report_error("Callback was never set.");
    }
    callback_->put_matrix(next_draw().to_matrix());
  }
  
  //======================================================================
  NativeArrayListElement::NativeArrayListElement(ArrayIoCallback *callback,
                                                 const std::string &name,
                                                 bool allow_streaming)
      : PythonListIoElement(name),
        callback_(callback),
        array_buffer_(NULL, std::vector<int>()),
        allow_streaming_(allow_streaming)
  {
    if (!callback) {
      report_error("NULL callback passed to NativeArrayListElement.");
    }
  }

  void NativeArrayListElement::prepare_to_write(int niter) {
    std::vector<int> dims = callback_->dim();
    std::vector<int> array_dims(dims.size() + 1);
    array_dims[0] = niter;
    std::copy(dims.begin(), dims.end(), array_dims.begin() + 1);
    int alloc_size = 1;
    for (int i = 0; i < array_dims.size(); ++i) {
      alloc_size *= array_dims[i];
    }

    internal_data_.resize(alloc_size);

    array_buffer_.reset(internal_data_.data(), array_dims);
    array_view_index_.assign(array_dims.size(), -1);
  }

  double * NativeArrayListElement::data() {
    return internal_data_.data();
  }

  void NativeArrayListElement::write() {
    ArrayView view(next_array_view());
    callback_->write_to_array(view);
  }

  void NativeArrayListElement::stream() {
    if (!allow_streaming_) return;
    ArrayView view(next_array_view());
    callback_->read_from_array(view);
  }

  ArrayView NativeArrayListElement::next_array_view() {
    array_view_index_[0] = next_position();
    return array_buffer_.slice(array_view_index_);
  }
  //======================================================================
  PythonListOfMatricesListElement::PythonListOfMatricesListElement(
      const std::string &name,
      const std::vector<int> &rows,
      const std::vector<int> &cols,
      Callback *callback)
      : PythonListIoElement(name),
        rows_(rows),
        cols_(cols),
        callback_(callback)
  {
    if (rows_.size() != cols_.size()) {
      report_error("The vectors listing the number of rows and columns in "
                   "the stored matrices must be the same size.");
    }
  }
  
  double *PythonListOfMatricesListElement::data() {
    return internal_data_.data();
  }

  void PythonListOfMatricesListElement::prepare_to_write(int niter) {
    int number_of_matrices = rows_.size();
    int alloc_size = 0;
    int cur_start_idx = 0;
    for (int i = 0; i < number_of_matrices; ++i) {
      alloc_size += niter * rows_[i] * cols_[i];
    }
    internal_data_.resize(alloc_size);
    views_.clear();
    for (int i = 0; i < number_of_matrices; ++i) {
      std::vector<int> array_dims = {niter, rows_[i], cols_[i]};
      views_.push_back(ArrayView(&data()[cur_start_idx], array_dims));
      cur_start_idx += niter * rows_[i] * cols_[i];
    }
  }

  void PythonListOfMatricesListElement::write() {
    int iteration = next_position();
    for (int layer = 0; layer < views_.size(); ++layer) {
      views_[layer].slice(iteration, -1, -1) = callback_->get(layer);
    }
  }

  void PythonListOfMatricesListElement::stream() {
    int iteration = next_position();
    for (int layer = 0; layer < views_.size(); ++layer) {
      callback_->put(layer, views_[layer].slice(iteration, -1, -1));
    }
  }

}  // namespace BOOM
