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
#include <Models/StateSpace/StateSpaceModelBase.hpp>
#include <functional>
#include <list>

//==============================================================================
// Note that the functions listed here throw exceptions.  Code that uses them
// should be wrapped in a try-block where the catch statement catches the
// exception and calls Rf_error() with an appropriate error message.  The
// functions handle_exception(), and handle_unknown_exception (in
// handle_exception.hpp), are suitable defaults.  These try-blocks should be
// present in any code called directly from R by .Call.
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

  namespace PythonInterface {
    class StateModelFactoryBase {
     public:
      StateModelFactoryBase()
      {}

      // Save the final state (i.e. at time T) of the model for use with
      // prediction.  Do not call this function until after all components of
      // state have been added.
      // Args:
      //   model:  A pointer to the model that owns the state.
      //   final_state: A pointer to a Vector to hold the state.  This can be
      //     nullptr if the state is only going to be recorded.  If state is
      //     going to be read, then final_state must be non-NULL.  A non-NULL
      //     vector will be re-sized if it is the wrong size.
      //   list_element_name: The name of the final state vector in the R list
      //     holding the MCMC output.
      void SaveFinalState(StateSpaceModelBase *model,
                          BOOM::Vector *final_state = nullptr,
                          const std::string & list_element_name = "final.state");
      
      const std::vector<int> DynamicRegressionStateModelPositions() const {
        return dynamic_regression_state_model_positions_;
      }
      
     protected:

     
     private:

    };

    //==========================================================================
    // A factory for creating state components for use with state space models.
    // This class can be used to add state to a ScalarStateSpaceModelBase or a
    // DynamicInterceptRegressionModel.  As new state space models are
    // developed, it can be extended by adding an AddState method appropriate
    // for the new model class.
    class StateModelFactory : public StateModelFactoryBase {
     public:
      // Args:
      //   io_manager: A pointer to the object manaaging the R list that will
      //     record (or has already recorded) the MCMC output.  If a nullptr is
      //     passed then states will be created without IoManager support.
      explicit StateModelFactory();

      // Adds all the state components listed in
      // r_state_specification_list to the model.
      // Args:
      //   model: The model to which the state will be added.  
      //   r_state_specification_list: An R list of state components to be added
      //     to the model.  This function intended to handle the state
      //     specification argument in bsts.
      //   prefix: An optional prefix added to the name of each state component.
      void AddState(ScalarStateSpaceModelBase *model,
                    ScalarStateSpaceSpecification *specification,
                    const std::string &prefix = "");
      void AddState(DynamicInterceptRegressionModel *model,
                    ScalarStateSpaceSpecification *specification
                    const std::string &prefix = "");
      
      // Factory method for creating a StateModel based on inputs supplied to R.
      // Returns a smart pointer to the StateModel that gets created.
      // Args:
      //   model: The state space model to which this state model will be added.
      //   r_state_component: The portion of the state.specification list (that
      //     was supplied to R by the user), corresponding to the state model
      //     that needs to be created.
      //   prefix: A prefix to be added to the name field of the
      //     r_state_component in the io_manager.
      // Returns:
      //   A Ptr to a StateModel that can be added as a component of state to a
      //   state space model.
      Ptr<StateModel> CreateStateModel(ScalarStateSpaceModelBase *model,
                                       ScalarStateSpaceSpecification *specification,
                                       const std::string &prefix);

      Ptr<DynamicInterceptStateModel> CreateDynamicInterceptStateModel(
          DynamicInterceptRegressionModel *model,
          ScalarStateSpaceSpecification *specification,
          const std::string &prefix);

      static Ptr<Holiday> CreateHoliday(SEXP holiday_spec);

     private:
      // Concrete implementations of CreateStateModel.

      LocalLevelStateModel *CreateLocalLevel(
          ScalarStateSpaceSpecification *specification, const std::string &prefix);
      LocalLinearTrendStateModel *CreateLocalLinearTrend(
          ScalarStateSpaceSpecification *specification, const std::string &prefix);
      SemilocalLinearTrendStateModel *CreateSemilocalLinearTrend(
          ScalarStateSpaceSpecification *specification, const std::string &prefix);
      StudentLocalLinearTrendStateModel *CreateStudentLocalLinearTrend(
          ScalarStateSpaceSpecification *specification, const std::string &prefix);
      StaticInterceptStateModel *CreateStaticIntercept(
          ScalarStateSpaceSpecification *specification, const std::string &prefix);
      ArStateModel *CreateArStateModel(
          ScalarStateSpaceSpecification *specification, const std::string &prefix);
      ArStateModel *CreateAutoArStateModel(
          ScalarStateSpaceSpecification *specification, const std::string &prefix);

      DynamicRegressionStateModel *CreateDynamicRegressionStateModel(
          ScalarStateSpaceSpecification *specification,
          const std::string &prefix,
          StateSpaceModelBase *model);
      DynamicRegressionArStateModel *CreateDynamicRegressionArStateModel(
          ScalarStateSpaceSpecification *specification,
          const std::string &prefix,
          StateSpaceModelBase *model);

      RandomWalkHolidayStateModel *CreateRandomWalkHolidayStateModel(
          ScalarStateSpaceSpecification *specification, const std::string &prefix);

      ScalarRegressionHolidayStateModel *CreateRegressionHolidayStateModel(
          ScalarStateSpaceSpecification *specification,
          const std::string &prefix,
          ScalarStateSpaceModelBase *model);
      DynamicInterceptRegressionHolidayStateModel *
      CreateDynamicInterceptRegressionHolidayStateModel(
          ScalarStateSpaceSpecification *specification,
          const std::string &prefix,
          DynamicInterceptRegressionModel *model);
      void ImbueRegressionHolidayStateModel(
          RegressionHolidayStateModel *holiday_model,
          ScalarStateSpaceSpecification *specification,
          const std::string &prefix);
      
      ScalarHierarchicalRegressionHolidayStateModel *
      CreateHierarchicalRegressionHolidayStateModel(
          ScalarStateSpaceSpecification *specification,
          const std::string &prefix,
          ScalarStateSpaceModelBase *model);
      DynamicInterceptHierarchicalRegressionHolidayStateModel *
      CreateDIHRHSM(ScalarStateSpaceSpecification *specification,
                    const std::string &prefix,
                    DynamicInterceptRegressionModel *model);
      void ImbueHierarchicalRegressionHolidayStateModel(
          HierarchicalRegressionHolidayStateModel *holiday_model,
          SEXP r_state_specification,
          const std::string &prefix);
      
      SeasonalStateModel *CreateSeasonal(
          ScalarStateSpaceSpecification *specification, const std::string &prefix);
      TrigStateModel *CreateTrigStateModel(
          ScalarStateSpaceSpecification *specification, const std::string &prefix);
      TrigRegressionStateModel *CreateTrigRegressionStateModel(
          ScalarStateSpaceSpecification *specification, const std::string &prefix);
      MonthlyAnnualCycle *CreateMonthlyAnnualCycle(
          ScalarStateSpaceSpecification *specification, const std::string &prefix);
    };

  }  // namespace PythonInterface
}  // namespace BOOM
#endif  // BOOM_PYTHON_INTERFACE_CREATE_STATE_MODEL_HPP_
