// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#ifndef BOOM_PYTHON_INTERFACE_CREATE_STATE_MODEL_HPP_
#define BOOM_PYTHON_INTERFACE_CREATE_STATE_MODEL_HPP_

#include "model_manager.hpp"
#include "list_io.hpp"
#include "state_space_regression_model_manager.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include <functional>
#include <list>

//==============================================================================
// Note that the functions listed here throw exceptions.  Code that uses them
// should be wrapped in a try-block where the catch statement catches the
// exception and calls report_error() with an appropriate error message.
// ==============================================================================

namespace BOOM {

  // Trend models
  class LocalLevelStateModel;
  class LocalLinearTrendStateModel;
  class SemilocalLinearTrendStateModel;
  class StudentLocalLinearTrendStateModel;
  class StaticInterceptStateModel;
  class ArStateModel;

  // Regression models
  class DynamicRegressionStateModel;
  class DynamicRegressionArStateModel;

  // Seasonal Models
  class MonthlyAnnualCycle;
  class SeasonalStateModel;
  class TrigStateModel;
  class TrigRegressionStateModel;

  // Holiday models
  class Holiday;
  class RandomWalkHolidayStateModel;

  class RegressionHolidayStateModel;
  class ScalarRegressionHolidayStateModel;
  class DynamicInterceptRegressionHolidayStateModel;

  class HierarchicalRegressionHolidayStateModel;
  class ScalarHierarchicalRegressionHolidayStateModel;
  class DynamicInterceptHierarchicalRegressionHolidayStateModel;

  namespace pybsts {
    class StateModelFactoryBase {
     public:
      explicit StateModelFactoryBase(std::shared_ptr<PythonListIoManager> io_manager)
        : io_manager_(io_manager)
      {}

      void SaveFinalState(StateSpaceModelBase *model,
                          BOOM::Vector *final_state = nullptr,
                          const std::string & list_element_name = "final.state");

      const std::vector<int> DynamicRegressionStateModelPositions() const {
        return dynamic_regression_state_model_positions_;
      }
      
     protected:
      void InstallPostStateListElements() {
        if (io_manager_) {
          for (int i = 0; i < post_state_list_elements_.size(); ++i) {
            io_manager_->add_list_element(post_state_list_elements_[i]);
          }
        }
        // The post state list elements will be empty after a call to this
        // function, whether or not io_manager_ is defined.
        post_state_list_elements_.clear();
      }

      void AddPostStateListElement(PythonListIoElement *element) {
        post_state_list_elements_.push_back(element);
      }

      void IdentifyDynamicRegression(int position) {
        dynamic_regression_state_model_positions_.push_back(position);
      }

      std::shared_ptr<PythonListIoManager> io_manager() {return io_manager_;}
     
     private:
      std::shared_ptr<PythonListIoManager> io_manager_;
      std::vector<PythonListIoElement *> post_state_list_elements_;
      std::vector<int> dynamic_regression_state_model_positions_;

    };

    //==========================================================================
    // A factory for creating state components for use with state space models.
    // This class can be used to add state to a ScalarStateSpaceModelBase or a
    // DynamicInterceptRegressionModel.  As new state space models are
    // developed, it can be extended by adding an AddState method appropriate
    // for the new model class.
    class StateModelFactory : public StateModelFactoryBase {
     public:
      explicit StateModelFactory(std::shared_ptr<PythonListIoManager> io_manager);

      void AddState(ScalarManagedModel *model,
                    const ScalarStateSpaceSpecification *specification,
                    const std::string &prefix);

      // static Ptr<Holiday> CreateHoliday(const ScalarSateSpaceSpecification *specification, const Holiday *holiday);
     protected:
      void IdentifyDynamicRegression(int position) {
        dynamic_regression_state_model_positions_.push_back(position);
      }


     private:
      LocalLevelStateModel *CreateLocalLevel(const ScalarStateSpaceSpecification *specification, const std::string &prefix);
      LocalLinearTrendStateModel *CreateLocalLinearTrend(const ScalarStateSpaceSpecification *specification, const std::string &prefix);
      SemilocalLinearTrendStateModel *CreateSemilocalLinearTrend(const ScalarStateSpaceSpecification *specification, const std::string &prefix);
      StudentLocalLinearTrendStateModel *CreateStudentLocalLinearTrend(const ScalarStateSpaceSpecification *specification, const std::string &prefix);
      StaticInterceptStateModel *CreateStaticIntercept(const ScalarStateSpaceSpecification *specification, const std::string &prefix);
      ArStateModel *CreateArStateModel(const ScalarStateSpaceSpecification *specification, const std::string &prefix);
      ArStateModel *CreateAutoArStateModel(const ScalarStateSpaceSpecification *specification, const std::string &prefix);

      DynamicRegressionStateModel *CreateDynamicRegressionStateModel(
          const ScalarStateSpaceSpecification *specification,
          StateSpaceRegressionManagedModel *model,
          const std::string &prefix);
      DynamicRegressionArStateModel *CreateDynamicRegressionArStateModel(
          const ScalarStateSpaceSpecification *specification,
          StateSpaceRegressionManagedModel *model,
          const std::string &prefix);

      // RandomWalkHolidayStateModel *CreateRandomWalkHolidayStateModel(const ScalarStateSpaceSpecification *specification, const std::string &prefix);

      // ScalarRegressionHolidayStateModel *CreateRegressionHolidayStateModel(
      //     const ScalarStateSpaceSpecification *specification,
      //     ScalarStateSpaceModelBase *model);
      // DynamicInterceptRegressionHolidayStateModel *
      // CreateDynamicInterceptRegressionHolidayStateModel(
      //     const ScalarStateSpaceSpecification *specification,
      //     DynamicInterceptRegressionModel *model);
      // void ImbueRegressionHolidayStateModel(
      //     RegressionHolidayStateModel *holiday_model,
      //     const ScalarStateSpaceSpecification *specification, const std::string &prefix);
      
      // ScalarHierarchicalRegressionHolidayStateModel *
      // CreateHierarchicalRegressionHolidayStateModel(
      //     const ScalarStateSpaceSpecification *specification,
      //     ScalarStateSpaceModelBase *model);
      // DynamicInterceptHierarchicalRegressionHolidayStateModel *
      // CreateDIHRHSM(const ScalarStateSpaceSpecification *specification,
      //               DynamicInterceptRegressionModel *model);
      // void ImbueHierarchicalRegressionHolidayStateModel(
      //     HierarchicalRegressionHolidayStateModel *holiday_model,
      //     const ScalarStateSpaceSpecification *specification, const std::string &prefix);
      
      SeasonalStateModel *CreateSeasonal(const ScalarStateSpaceSpecification *specification,
        SeasonSpecification* season,
        const std::string &prefix);
      // TrigStateModel *CreateTrigStateModel(const ScalarStateSpaceSpecification *specification, const std::string &prefix);
      // TrigRegressionStateModel *CreateTrigRegressionStateModel(const ScalarStateSpaceSpecification *specification, const std::string &prefix);
      // MonthlyAnnualCycle *CreateMonthlyAnnualCycle(const ScalarStateSpaceSpecification *specification, const std::string &prefix);

      // The index of each dynamic regression state model, in the vector of
      // state models held by the main state space model.
      std::vector<int> dynamic_regression_state_model_positions_;
    };

  }  // namespace pybsts
}  // namespace BOOM
#endif  // BOOM_PYTHON_INTERFACE_CREATE_STATE_MODEL_HPP_
