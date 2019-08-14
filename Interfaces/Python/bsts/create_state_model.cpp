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

#include "create_state_model.hpp"
#include <string>
#include "cpputil/report_error.hpp"
#include "cpputil/Date.hpp"

#include "prior_specification.hpp"

#include "Models/ChisqModel.hpp"
#include "Models/GaussianModel.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"
#include "Models/PosteriorSamplers/FixedSpdSampler.hpp"
#include "Models/PosteriorSamplers/FixedUnivariateSampler.hpp"
#include "Models/PosteriorSamplers/GammaPosteriorSampler.hpp"
#include "Models/PosteriorSamplers/IndependentMvnVarSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanMvnIndependenceSampler.hpp"

#include "Models/Hierarchical/PosteriorSamplers/HierGaussianRegressionAsisSampler.hpp"

#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "Models/StateSpace/DynamicInterceptRegression.hpp"

#include "Models/StateSpace/PosteriorSamplers/DynamicRegressionArPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/DynamicRegressionPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StudentLocalLinearTrendPosteriorSampler.hpp"
#include "Models/StateSpace/StateModels/ArStateModel.hpp"
#include "Models/StateSpace/StateModels/DynamicRegressionArStateModel.hpp"
#include "Models/StateSpace/StateModels/DynamicRegressionStateModel.hpp"
#include "Models/StateSpace/StateModels/HierarchicalRegressionHolidayStateModel.hpp"
#include "Models/StateSpace/StateModels/Holiday.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateModels/LocalLinearTrend.hpp"
#include "Models/StateSpace/StateModels/RandomWalkHolidayStateModel.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"
#include "Models/StateSpace/StateModels/SemilocalLinearTrend.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/StateSpace/StateModels/StaticInterceptStateModel.hpp"
#include "Models/StateSpace/StateModels/StudentLocalLinearTrend.hpp"
#include "Models/StateSpace/StateModels/TrigStateModel.hpp"

#include "Models/TimeSeries/NonzeroMeanAr1Model.hpp"
#include "Models/TimeSeries/PosteriorSamplers/NonzeroMeanAr1Sampler.hpp"
#include "Models/TimeSeries/PosteriorSamplers/ArPosteriorSampler.hpp"
#include "Models/TimeSeries/PosteriorSamplers/ArSpikeSlabSampler.hpp"

namespace BOOM {
  namespace pybsts {

    StateModelFactory::StateModelFactory(std::shared_ptr<PythonListIoManager> io_manager) :
      StateModelFactoryBase(io_manager)
    {}

    void StateModelFactory::AddState(ScalarManagedModel *model,
                                     const ScalarStateSpaceSpecification *specification,
                                     const std::string &prefix) {
      if ((!model)||(!model->sampling_model())) return;
      if (specification->local_trend()) {
        if (specification->local_trend()->trend_prior()) {
          if (specification->local_trend()->slope_prior()) {
            if (specification->local_trend()->slope_bias_prior()) {
              model->sampling_model()->add_state(CreateSemilocalLinearTrend(specification, prefix));
            } else {
              if (specification->local_trend()->student_errors()) {
                model->sampling_model()->add_state(CreateStudentLocalLinearTrend(specification, prefix));
              } else {
                model->sampling_model()->add_state(CreateLocalLinearTrend(specification, prefix));
              }
            }
          } else {
            model->sampling_model()->add_state(CreateLocalLevel(specification, prefix));
          }
        } else if (specification->local_trend()->static_intercept()) {
          model->sampling_model()->add_state(CreateStaticIntercept(specification, prefix));
        }
      }
      if (!specification->dynamic_regression()) {
        if (specification->ar_prior()) {
          model->sampling_model()->add_state(CreateAutoArStateModel(specification, prefix));
        } else if (specification->ar_order() > 0) {
          model->sampling_model()->add_state(CreateArStateModel(specification, prefix));
        }
      } else if ((specification->ar_order() > 0) && (specification->dynamic_regression())) {
        model->sampling_model()->add_state(CreateDynamicRegressionArStateModel(specification, dynamic_cast<StateSpaceRegressionManagedModel*>(model), prefix));
      } else if (specification->dynamic_regression()) {
         model->sampling_model()->add_state(CreateDynamicRegressionStateModel(specification, dynamic_cast<StateSpaceRegressionManagedModel*>(model), prefix));
      }
      for (std::size_t i = 0; i < (specification->seasons()).size(); ++i) {
        model->sampling_model()->add_state(CreateSeasonal(specification, &(specification->seasons()[i]), prefix));
      }
      InstallPostStateListElements();      
    }

    // void StateModelFactory::AddState(ScalarManagedModel *model,
    //                                  const ScalarStateSpaceSpecification *specification,
    //                                  const std::string &prefix) {
    //   if (!model) return;
    //   if (specification->local_trend()) {
    //     if (specification->local_trend()->trend_prior()) {
    //       if (specification->local_trend()->slope_prior()) {
    //         if (specification->local_trend()->slope_bias_prior()) {
    //           model->add_state(new DynamicInterceptStateModelAdapter(CreateSemilocalLinearTrend(specification)));
    //         } else {
    //           if (specification->local_trend()->student_errors()) {
    //             model->add_state(new DynamicInterceptStateModelAdapter(CreateStudentLocalLinearTrend(specification)));
    //           } else {
    //             model->add_state(new DynamicInterceptStateModelAdapter(CreateLocalLinearTrend(specification)));
    //           }
    //         }
    //       } else {
    //         model->add_state(new DynamicInterceptStateModelAdapter(CreateLocalLevel(specification)));
    //       }
    //     } else if (specification->local_trend()->static_intercept()) {
    //       model->add_state(new DynamicInterceptStateModelAdapter(CreateStaticIntercept(specification)));
    //     }
    //   }
    //   if (!specification->dynamic_regression()) {
    //     if (specification->ar_prior()) {
    //       model->add_state(new DynamicInterceptStateModelAdapter(CreateAutoArStateModel(specification)));
    //     } else if (specification->ar_order() > 0) {
    //       model->add_state(new DynamicInterceptStateModelAdapter(CreateArStateModel(specification)));
    //     }
    //   } else if ((specification->ar_order() > 0) && (specification->dynamic_regression())) {
    //     model->add_state(new DynamicInterceptStateModelAdapter(CreateDynamicRegressionArStateModel(specification, model)));
    //   } else if (specification->dynamic_regression()) {
    //      model->add_state(new DynamicInterceptStateModelAdapter(CreateDynamicRegressionStateModel(specification, model)));
    //   }
    //   for (auto season = specification->seasons().begin(); season != specification->seasons().end(); season++) {
    //     model->add_state(new DynamicInterceptStateModelAdapter(CreateSeasonal(specification, season)));
    //   }
    // }

    // A factory function that unpacks information from an R object created by
    // AddXXX (where XXX is the name of a type of state model), and use it to
    // build the appropriate BOOM StateModel.  The specific R function
    // associated with each method is noted in the comments to the worker
    // functions that implement each specific type.
    // Args:
    //   r_state_component:  The R object created by AddXXX.
    //   prefix: An optional prefix to be prepended to the name of the state
    //     component in the io_manager.
    // Returns:
    //   A BOOM smart pointer to the appropriately typed StateModel.
    // Ptr<StateModel> StateModelFactory::CreateStateModel(
    //     ScalarStateSpaceModelBase *model,
    //     const ScalarStateSpaceSpecification *specification) {
    //   if (Rf_inherits(r_state_component, "AutoAr")) {
    //     // AutoAr also inherits from ArProcess, so this case must be
    //     // handled before ArProcess.
    //     return CreateAutoArStateModel(r_state_component, prefix);
    //   } else if (Rf_inherits(r_state_component, "ArProcess")) {
    //     return CreateArStateModel(r_state_component, prefix);
    //   } else if (Rf_inherits(r_state_component, "DynamicRegression")) {
    //     SEXP r_model_options = getListElement(
    //         r_state_component, "model.options");
    //     if (Rf_inherits(
    //             r_model_options, "DynamicRegressionRandomWalkOptions")) {
    //       return CreateDynamicRegressionStateModel(
    //           r_state_component, prefix, model);
    //     } else if (Rf_inherits(
    //         r_model_options, "DynamicRegressionArOptions")) {
    //       return CreateDynamicRegressionArStateModel(
    //           r_state_component, prefix, model);
    //     } else {
    //       report_error("Unrecognized 'model.options' object in dynamic "
    //                    "regression state component.");
    //       return Ptr<StateModel>(nullptr);
    //     }
    //   } else if (Rf_inherits(r_state_component, "LocalLevel")) {
    //     return CreateLocalLevel(r_state_component, prefix);
    //   } else if (Rf_inherits(r_state_component, "LocalLinearTrend")) {
    //     return CreateLocalLinearTrend(r_state_component, prefix);
    //   } else if (Rf_inherits(r_state_component, "Monthly")) {
    //     return CreateMonthlyAnnualCycle(r_state_component, prefix);
    //   } else if (Rf_inherits(r_state_component, "Seasonal")) {
    //     return CreateSeasonal(r_state_component, prefix);
    //   } else if (Rf_inherits(r_state_component, "SemilocalLinearTrend")) {
    //     return CreateSemilocalLinearTrend(r_state_component, prefix);
    //   } else if (Rf_inherits(r_state_component, "StaticIntercept")) {
    //     return CreateStaticIntercept(r_state_component, prefix);
    //   } else if (Rf_inherits(r_state_component, "StudentLocalLinearTrend")) {
    //     return CreateStudentLocalLinearTrend(r_state_component, prefix);
    //   } else if (Rf_inherits(r_state_component, "Trig")) {
    //     std::string method = ToString(getListElement(
    //         r_state_component, "method", true));
    //     if (method == "direct") {
    //       return CreateTrigRegressionStateModel(r_state_component, prefix);
    //     } else if (method == "harmonic") {
    //       return CreateTrigStateModel(r_state_component, prefix);
    //     } else {
    //       std::ostringstream err;
    //       err << "Unknown method: " << method
    //           << " in state specification for trig state model.";
    //       report_error(err.str());
    //       return nullptr;
    //     }
    //   } else if (Rf_inherits(r_state_component, "RandomWalkHolidayStateModel")) {
    //     return CreateRandomWalkHolidayStateModel(r_state_component, prefix);
    //   } else if (Rf_inherits(
    //       r_state_component, "HierarchicalRegressionHolidayStateModel")) {
    //     return CreateHierarchicalRegressionHolidayStateModel(
    //         r_state_component, prefix, model);
    //   } else if (Rf_inherits(r_state_component, "RegressionHolidayStateModel")) {
    //     return CreateRegressionHolidayStateModel(r_state_component, prefix, model);
    //   } else {
    //     std::ostringstream err;
    //     err << "Unknown object passed where state model expected." << endl;
    //     std::vector<std::string> class_info = StringVector(
    //         Rf_getAttrib(r_state_component, R_ClassSymbol));
    //     if (class_info.empty()) {
    //       err << "Object has no class attribute." << endl;
    //     } else if (class_info.size() == 1) {
    //       err << "Object is of class " << class_info[0] << "." << endl;
    //     } else {
    //       err << "Object has class:" << endl;
    //       for (int i = 0; i < class_info.size(); ++i) {
    //         err << "     " << class_info[i] << endl;
    //       }
    //       report_error(err.str());
    //     }
    //     return nullptr;
    //   }
    // }

    // A factory function that unpacks information from an R object created by
    // AddXXX (where XXX is the name of a type of state model), and use it to
    // build the appropriate BOOM StateModel.  The specific R function
    // associated with each method is noted in the comments to the worker
    // functions that implement each specific type.
    // Args:
    //   r_state_component:  The R object created by AddXXX.
    //   prefix: An optional prefix to be prepended to the name of the state
    //     component in the io_manager.
    // Returns:
    //   A BOOM smart pointer to the appropriately typed StateModel.
    // Ptr<DynamicInterceptStateModel>
    // StateModelFactory::CreateDynamicInterceptStateModel(
    //     DynamicInterceptRegressionModel *model,
    //     const ScalarStateSpaceSpecification *specification) {
    //   if (Rf_inherits(r_state_component, "AutoAr")) {
    //     // AutoAr also inherits from ArProcess, so this case must be
    //     // handled before ArProcess.
    //     return new DynamicInterceptStateModelAdapter(
    //         CreateAutoArStateModel(r_state_component, prefix));
    //   } else if (Rf_inherits(r_state_component, "ArProcess")) {
    //     return new DynamicInterceptStateModelAdapter(
    //         CreateArStateModel(r_state_component, prefix));
    //   } else if (Rf_inherits(r_state_component, "DynamicRegression")) {
    //     SEXP r_model_options = getListElement(
    //         r_state_component, "model.options");
    //     if (Rf_inherits(
    //             r_model_options, "DynamicRegressionRandomWalkOptions")) {
    //       return new DynamicInterceptStateModelAdapter(
    //           CreateDynamicRegressionStateModel(r_state_component, prefix, model));
    //     } else if (Rf_inherits(
    //         r_model_options, "DynamicRegressionArOptions")) {
    //       return new DynamicInterceptStateModelAdapter(
    //           CreateDynamicRegressionArStateModel(r_state_component, prefix, model));
    //     } else {
    //       report_error("Unrecognized 'model.options' object in dynamic "
    //                    "regression state component.");
    //       return Ptr<DynamicInterceptStateModel>(nullptr);
    //     }
    //   } else if (Rf_inherits(r_state_component, "LocalLevel")) {
    //     return new DynamicInterceptStateModelAdapter(
    //         CreateLocalLevel(r_state_component, prefix));
    //   } else if (Rf_inherits(r_state_component, "LocalLinearTrend")) {
    //     return new DynamicInterceptStateModelAdapter(
    //         CreateLocalLinearTrend(r_state_component, prefix));
    //   } else if (Rf_inherits(r_state_component, "Monthly")) {
    //     return new DynamicInterceptStateModelAdapter(
    //         CreateMonthlyAnnualCycle(r_state_component, prefix));
    //   } else if (Rf_inherits(r_state_component, "Seasonal")) {
    //     return new DynamicInterceptStateModelAdapter(
    //         CreateSeasonal(r_state_component, prefix));
    //   } else if (Rf_inherits(r_state_component, "SemilocalLinearTrend")) {
    //     return new DynamicInterceptStateModelAdapter(
    //         CreateSemilocalLinearTrend(r_state_component, prefix));
    //   } else if (Rf_inherits(r_state_component, "StaticIntercept")) {
    //     return new DynamicInterceptStateModelAdapter(
    //         CreateStaticIntercept(r_state_component, prefix));
    //   } else if (Rf_inherits(r_state_component, "StudentLocalLinearTrend")) {
    //     return new DynamicInterceptStateModelAdapter(
    //         CreateStudentLocalLinearTrend(r_state_component, prefix));
    //   } else if (Rf_inherits(r_state_component, "Trig")) {
    //     std::string method = ToString(getListElement(
    //         r_state_component, "method", true));
    //     if (method == "direct") {
    //       return new DynamicInterceptStateModelAdapter(
    //           CreateTrigRegressionStateModel(r_state_component, prefix));
    //     } else if (method == "harmonic") {
    //       return new DynamicInterceptStateModelAdapter(
    //           CreateTrigStateModel(r_state_component, prefix));
    //     } else {
    //       std::ostringstream err;
    //       err << "Unknown method: " << method
    //           << " in state specification for trig state model.";
    //       report_error(err.str());
    //       return nullptr;
    //     }
    //   } else if (Rf_inherits(r_state_component, "RandomWalkHolidayStateModel")) {
    //     return new DynamicInterceptStateModelAdapter(
    //         CreateRandomWalkHolidayStateModel(r_state_component, prefix));
    //   } else if (Rf_inherits(
    //       r_state_component, "HierarchicalRegressionHolidayStateModel")) {
    //     return CreateDIHRHSM(r_state_component, prefix, model);
    //   } else if (Rf_inherits(r_state_component, "RegressionHolidayStateModel")) {
    //     return CreateDynamicInterceptRegressionHolidayStateModel(
    //         r_state_component, prefix, model);
    //   } else {
    //     std::ostringstream err;
    //     err << "Unknown object passed where dynamic intercept state model "
    //         << "expected." << endl;
    //     std::vector<std::string> class_info = StringVector(
    //         Rf_getAttrib(r_state_component, R_ClassSymbol));
    //     if (class_info.empty()) {
    //       err << "Object has no class attribute." << endl;
    //     } else if (class_info.size() == 1) {
    //       err << "Object is of class " << class_info[0] << "." << endl;
    //     } else {
    //       err << "Object has class:" << endl;
    //       for (int i = 0; i < class_info.size(); ++i) {
    //         err << "     " << class_info[i] << endl;
    //       }
    //       report_error(err.str());
    //     }
    //   }
    //   return nullptr;
    // }

    // A callback class for recording the final state that the
    // ScalarStateSpaceModelBase sampled in an MCMC iteration.
    class FinalStateCallback : public VectorIoCallback {
     public:
      explicit FinalStateCallback(StateSpaceModelBase *model)
          : model_(model) {}
      virtual int dim() const {return model_->state_dimension();}
      virtual Vector get_vector() const { return model_->final_state();}
     private:
      StateSpaceModelBase * model_;
    };

    void StateModelFactoryBase::SaveFinalState(
        StateSpaceModelBase *model,
        Vector * final_state,
        const std::string & list_element_name) {
      if (!model) return;
      if (final_state) {
        final_state->resize(model->state_dimension());
      }
      if (io_manager()) {
        io_manager()->add_list_element(
            new NativeVectorListElement(
                new BOOM::pybsts::FinalStateCallback(model),
                list_element_name,
                final_state));
      }
    }
    
    //======================================================================
    LocalLevelStateModel * StateModelFactory::CreateLocalLevel(
        const ScalarStateSpaceSpecification *specification, const std::string &prefix) {
      if (!(specification->local_trend())) {
        report_error("CreateLocalLevel: empty local_trend");
      }
      std::shared_ptr<PriorSpecification> trend_prior = specification->local_trend()->trend_prior();
      if (!trend_prior) {
        report_error("CreateLocalLevel: empty local_trend.trend_prior");
      }
      LocalLevelStateModel * level(
          new LocalLevelStateModel(trend_prior->prior_guess()));

      level->set_initial_state_variance(square(trend_prior->sigma_guess()));
      level->set_initial_state_mean(trend_prior->mu());

      if (trend_prior->fixed()) {
        Ptr<FixedUnivariateSampler> sampler(
            new FixedUnivariateSampler(
                level->Sigsq_prm(),
                level->sigsq()));
      } else {
        Ptr<ZeroMeanGaussianConjSampler> sampler(
            new ZeroMeanGaussianConjSampler(level,
                                            trend_prior->prior_df(),
                                            trend_prior->prior_guess()));
        if (trend_prior->sigma_upper_limit() > 0) {
          sampler->set_sigma_upper_limit(trend_prior->sigma_upper_limit());
        }
        level->set_method(sampler);
      }

      // Add information about this parameter to the io_manager
      if (io_manager()) {
        io_manager()->add_list_element(new StandardDeviationListElement(
            level->Sigsq_prm(),
            prefix + "sigma.level"));
      }
      return level;
    }

    LocalLinearTrendStateModel * StateModelFactory::CreateLocalLinearTrend(
        const ScalarStateSpaceSpecification *specification, const std::string &prefix) {

      LocalLinearTrendStateModel * local_linear_trend(
          new LocalLinearTrendStateModel);

      if (!(specification->local_trend())) {
        report_error("CreateLocalLinearTrend: empty local_trend");
      }
      std::shared_ptr<PriorSpecification> level_prior_spec = specification->local_trend()->trend_prior();
      std::shared_ptr<PriorSpecification> slope_prior_spec = specification->local_trend()->slope_prior();
      if (!level_prior_spec) {
        report_error("CreateLocalLinearTrend: empty local_trend.trend_prior");
      }
      if (!slope_prior_spec) {
        report_error("CreateLocalLinearTrend: empty local_trend.slope_prior");
      }

      Vector initial_state_mean(2);
      initial_state_mean[0] = level_prior_spec->mu();
      initial_state_mean[1] = slope_prior_spec->mu();
      local_linear_trend->set_initial_state_mean(initial_state_mean);

      SpdMatrix initial_state_variance(2);
      initial_state_variance(0, 0) =
          square(level_prior_spec->sigma_guess());
      initial_state_variance(1, 1) =
          square(slope_prior_spec->sigma_guess());
      local_linear_trend->set_initial_state_variance(initial_state_variance);

      // Set initial values of model parameters
      SpdMatrix Sigma = local_linear_trend->Sigma();
      Sigma(0, 0) = square(level_prior_spec->sigma_guess());
      Sigma(1, 1) = square(slope_prior_spec->sigma_guess());
      Sigma(0, 1) = 0;
      Sigma(1, 0) = 0;
      local_linear_trend->set_Sigma(Sigma);

      //----------------------------------------------------------------------
      // Set prior distribution for level_sigma.
      int pos = 0;
      if (level_prior_spec->fixed()) {
        Ptr<FixedSpdSampler> sampler(
            new FixedSpdSampler(local_linear_trend->Sigma_prm(),
                                square(level_prior_spec->sigma_guess()),
                                pos));
        local_linear_trend->set_method(sampler);
      } else {
        Ptr<ZeroMeanMvnIndependenceSampler> sampler(
            new ZeroMeanMvnIndependenceSampler(
                local_linear_trend,
                level_prior_spec->prior_df(),
                level_prior_spec->prior_guess(),
                pos));
        if (level_prior_spec->sigma_upper_limit() > 0) {
          sampler->set_sigma_upper_limit(level_prior_spec->sigma_upper_limit());
        }
        local_linear_trend->set_method(sampler);
      }

      //----------------------------------------------------------------------
      // Set prior distribution for slope_sigma.
      pos = 1;
      if (slope_prior_spec->fixed()) {
        Ptr<FixedSpdSampler> sampler(
            new FixedSpdSampler(local_linear_trend->Sigma_prm(),
                                square(slope_prior_spec->initial_value()),
                                pos));
        local_linear_trend->set_method(sampler);
      } else {
        Ptr<ZeroMeanMvnIndependenceSampler> sampler(
            new ZeroMeanMvnIndependenceSampler(
                local_linear_trend,
                slope_prior_spec->prior_df(),
                slope_prior_spec->prior_guess(),
                pos));
        if (slope_prior_spec->sigma_upper_limit() > 0) {
          sampler->set_sigma_upper_limit(slope_prior_spec->sigma_upper_limit());
        }
        local_linear_trend->set_method(sampler);
      }

      //----------------------------------------------------------------------
      // Now that the priors are all set, the last thing to do is to add the
      // model parameters to the io_manager.
      if (io_manager()) {
        io_manager()->add_list_element(
            new PartialSpdListElement(
                local_linear_trend->Sigma_prm(),
                prefix + "sigma.trend.level",
                0,
                true));

        io_manager()->add_list_element(
            new PartialSpdListElement(
                local_linear_trend->Sigma_prm(),
                prefix + "sigma.trend.slope",
                1,
                true));
      }
      return local_linear_trend;
    }

    //======================================================================
    // Two callback classes for recording the latent weights in the
    // StudentLocalLinearTrend state model.

    // Record weights for the level.
    class StudentLocalLinearTrendLevelWeightCallback
        : public VectorIoCallback {
     public:
      explicit StudentLocalLinearTrendLevelWeightCallback(
          StudentLocalLinearTrendStateModel *model) : model_(model) {}
      virtual int dim() const { return model_->latent_level_weights().size(); }
      virtual Vector get_vector() const {
        return model_->latent_level_weights();
      }
     private:
      StudentLocalLinearTrendStateModel *model_;
    };

    class StudentLocalLinearTrendSlopeWeightCallback
        : public VectorIoCallback {
     public:
      explicit StudentLocalLinearTrendSlopeWeightCallback(
          StudentLocalLinearTrendStateModel *model) : model_(model) {}
      virtual int dim() const { return model_->latent_level_weights().size(); }
      virtual Vector get_vector() const {
        return model_->latent_slope_weights();
      }
     private:
      StudentLocalLinearTrendStateModel *model_;
    };

    StudentLocalLinearTrendStateModel *
    StateModelFactory::CreateStudentLocalLinearTrend(const ScalarStateSpaceSpecification *specification,
        const std::string &prefix) {

      StudentLocalLinearTrendStateModel * robust_local_linear_trend(
          new StudentLocalLinearTrendStateModel(1, 10, 1, 10));

      if (!(specification->local_trend())) {
        report_error("CreateStudentLocalLinearTrend: empty local_trend");
      }

      //----------------------------------------------------------------------
      // Unpack the prior and create the posterior sampler.
      std::shared_ptr<PriorSpecification> level_prior_spec = specification->local_trend()->trend_prior();
      std::shared_ptr<PriorSpecification> slope_prior_spec = specification->local_trend()->slope_prior();
      if (!level_prior_spec) {
        report_error("CreateStudentLocalLinearTrend: empty local_trend.trend_prior");
      }
      if (!slope_prior_spec) {
        report_error("CreateStudentLocalLinearTrend: empty local_trend.slope_prior");
      }

      NEW(ChisqModel, level_sigma_prior)(
          level_prior_spec->prior_df(),
          level_prior_spec->prior_guess());
      NEW(ChisqModel, slope_sigma_prior)(
          slope_prior_spec->prior_df(),
          slope_prior_spec->prior_guess());

      if (!specification->local_trend()->trend_df_prior()) {
        report_error("CreateStudentLocalLinearTrend: empty local_trend.trend_df_prior");
      }
      if (!specification->local_trend()->slope_df_prior()) {
        report_error("CreateStudentLocalLinearTrend: empty local_trend.slope_df_prior");
      }

      Ptr<DoubleModel> level_nu_prior(create_double_model(specification->local_trend()->trend_df_prior()));
      Ptr<DoubleModel> slope_nu_prior(create_double_model(specification->local_trend()->slope_df_prior()));

      NEW(StudentLocalLinearTrendPosteriorSampler, sampler)(
          robust_local_linear_trend,
          level_sigma_prior,
          level_nu_prior,
          slope_sigma_prior,
          slope_nu_prior);
      sampler->set_sigma_slope_upper_limit(
          slope_prior_spec->sigma_upper_limit());
      sampler->set_sigma_level_upper_limit(
          level_prior_spec->sigma_upper_limit());
      robust_local_linear_trend->set_method(sampler);

      //----------------------------------------------------------------------
      // Set the prior for the initial state.
      Vector initial_state_mean(2);
      initial_state_mean[0] = level_prior_spec->mu();
      initial_state_mean[1] = slope_prior_spec->mu();
      robust_local_linear_trend->set_initial_state_mean(initial_state_mean);

      SpdMatrix initial_state_variance(2);
      initial_state_variance(0, 0) =
          square(level_prior_spec->sigma_guess());
      initial_state_variance(1, 1) =
          square(slope_prior_spec->sigma_guess());
      robust_local_linear_trend->set_initial_state_variance(
          initial_state_variance);

      //----------------------------------------------------------------------
      // Add parameters to io_manager.
      if (io_manager()) {
        io_manager()->add_list_element(
            new StandardDeviationListElement(
                robust_local_linear_trend->SigsqLevel_prm(),
                prefix + "sigma.trend.level"));
        io_manager()->add_list_element(
            new StandardDeviationListElement(
                robust_local_linear_trend->SigsqSlope_prm(),
                prefix + "sigma.trend.slope"));
        io_manager()->add_list_element(
            new UnivariateListElement(
                robust_local_linear_trend->NuLevel_prm(),
                prefix + "nu.trend.level"));
        io_manager()->add_list_element(
            new UnivariateListElement(
                robust_local_linear_trend->NuSlope_prm(),
                prefix + "nu.trend.slope"));
        io_manager()->add_list_element(
            new NativeVectorListElement(
                new StudentLocalLinearTrendLevelWeightCallback(
                    robust_local_linear_trend),
                prefix + "trend.level.weights",
                NULL));

        io_manager()->add_list_element(
            new NativeVectorListElement(
                new StudentLocalLinearTrendSlopeWeightCallback(
                    robust_local_linear_trend),
                prefix + "trend.slope.weights",
                NULL));
      }

      return robust_local_linear_trend;
    }
    //======================================================================
    StaticInterceptStateModel *StateModelFactory::CreateStaticIntercept(const ScalarStateSpaceSpecification *specification, 
        const std::string &prefix) {
      StaticInterceptStateModel *intercept = new StaticInterceptStateModel;
      std::shared_ptr<PriorSpecification> initial_state_prior = specification->initial_state_prior();
      if (!initial_state_prior) {
        report_error("CreateStaticIntercept: empty initial_state_prior");
      }
      intercept->set_initial_state_mean(initial_state_prior->mu());
      intercept->set_initial_state_variance(initial_state_prior->sigma_guess());
      return intercept;
    }
    // //======================================================================
    // TrigRegressionStateModel *StateModelFactory::CreateTrigRegressionStateModel(const ScalarStateSpaceSpecification *specification) {
    //   double period = Rf_asReal(getListElement(r_state_component, "period"));
    //   Vector frequencies = ToBoomVector(getListElement(
    //       r_state_component, "frequencies"));
    //   TrigRegressionStateModel * trig_state_model(
    //       new TrigRegressionStateModel(period, frequencies));

    //   //-------------- set the prior and the posterior sampler.
    //   SdPrior sigma_prior(getListElement(r_state_component, "sigma.prior"));
    //   int dimension = trig_state_model->dim();
    //   NEW(ChisqModel, single_siginv_prior)(
    //       sigma_prior.prior_df(),
    //       sigma_prior.prior_guess());
    //   std::vector<Ptr<GammaModelBase>> priors(dimension, single_siginv_prior);
    //   double sigma_upper_limit = sigma_prior.upper_limit();
    //   if (sigma_upper_limit < 0) {
    //     sigma_upper_limit = infinity();
    //   }
    //   Vector sd_max_values(dimension, sigma_upper_limit);
    //   NEW(IndependentMvnVarSampler, sampler)(
    //       trig_state_model,
    //       priors,
    //       sd_max_values);
    //   trig_state_model->set_method(sampler);

    //   //-------------- set the prior for the initial state
    //   MvnPrior initial_prior(getListElement(
    //       r_state_component, "initial.state.prior"));
    //   trig_state_model->set_initial_state_mean(initial_prior.mu());
    //   trig_state_model->set_initial_state_variance(initial_prior.Sigma());

    //   //-------------- adjust the io manager.
    //   if (io_manager()) {
    //     io_manager()->add_list_element(
    //         new SdVectorListElement(trig_state_model->Sigsq_prm(),
    //                                 prefix + "trig.coefficient.sd"));
    //   }
    //   return trig_state_model;
    // }
    // //======================================================================
    // TrigStateModel * StateModelFactory::CreateTrigStateModel(const ScalarStateSpaceSpecification *specification) {
    //   double period = Rf_asReal(getListElement(r_state_component, "period"));
    //   Vector frequencies = ToBoomVector(getListElement(
    //       r_state_component, "frequencies"));
    //   TrigStateModel * quasi_trig_state_model(
    //       new TrigStateModel(period, frequencies));

    //   //-------------- set the prior and the posterior sampler.
    //   SdPrior sigma_prior(getListElement(r_state_component, "sigma.prior"));
    //   NEW(ChisqModel, innovation_precision_prior)(
    //       sigma_prior.prior_df(),
    //       sigma_prior.prior_guess());
    //   double sigma_upper_limit = sigma_prior.upper_limit();
    //   if (sigma_upper_limit < 0) {
    //     sigma_upper_limit = infinity();
    //   }
    //   NEW(ZeroMeanGaussianConjSampler, error_distribution_sampler)(
    //       quasi_trig_state_model->error_distribution(),
    //       innovation_precision_prior);
    //   error_distribution_sampler->set_sigma_upper_limit(sigma_upper_limit);
    //   quasi_trig_state_model->set_method(error_distribution_sampler);

    //   //--------------- Set prior for initial state
    //   MvnPrior initial_prior(getListElement(
    //       r_state_component, "initial.state.prior", true));
    //   quasi_trig_state_model->set_initial_state_mean(initial_prior.mu());
    //   quasi_trig_state_model->set_initial_state_variance(initial_prior.Sigma());

    //   //--------------- Adjust the IO manager.
    //   if (io_manager()) {
    //     io_manager()->add_list_element(
    //         new StandardDeviationListElement(
    //             quasi_trig_state_model->error_distribution()->Sigsq_prm(),
    //             prefix + "trig.coefficient.sd"));
    //   }
    //   return quasi_trig_state_model;
    // }
    //======================================================================
    SemilocalLinearTrendStateModel *
    StateModelFactory::CreateSemilocalLinearTrend(const ScalarStateSpaceSpecification *specification,
      const std::string &prefix) {

      if (!(specification->local_trend())) {
        report_error("CreateSemilocalLinearTrend: empty local_trend");
      }

      std::shared_ptr<PriorSpecification> level_prior_spec = specification->local_trend()->trend_prior();
      if (!level_prior_spec) {
        report_error("CreateSemilocalLinearTrend: empty local_trend.trend_prior");
      }
      NEW(ZeroMeanGaussianModel, level)(level_prior_spec->prior_guess());

      std::shared_ptr<PriorSpecification> slope_mean_prior_spec = specification->local_trend()->slope_bias_prior();
      std::shared_ptr<PriorSpecification> slope_ar1_prior_spec = specification->local_trend()->slope_ar1_prior();
      std::shared_ptr<PriorSpecification> slope_prior_spec = specification->local_trend()->slope_prior();
      if (!slope_mean_prior_spec) {
        report_error("CreateSemilocalLinearTrend: empty local_trend.slope_bias_prior");
      }
      if (!slope_ar1_prior_spec) {
        report_error("CreateSemilocalLinearTrend: empty local_trend.slope_ar1_prior");
      }
      if (!slope_prior_spec) {
        report_error("CreateSemilocalLinearTrend: empty local_trend.slope_prior");
      }
      
      NEW(NonzeroMeanAr1Model, slope)(slope_mean_prior_spec->mu(),
                                      slope_ar1_prior_spec->mu(),
                                      slope_prior_spec->sigma_guess());

      SemilocalLinearTrendStateModel *trend
          = new SemilocalLinearTrendStateModel(level, slope);

      // Create the prior for level model.  This prior is simple,
      // because it is for a random walk.
      if (!level_prior_spec->fixed()) {
        NEW(ZeroMeanGaussianConjSampler, level_sampler)(
            level.get(),
            level_prior_spec->prior_df(),
            level_prior_spec->prior_guess());

        if (level_prior_spec->sigma_upper_limit() > 0) {
          level_sampler->set_sigma_upper_limit(
              level_prior_spec->sigma_upper_limit());
        }
        trend->set_method(level_sampler);
      }

      // Now create the prior for the slope model.  The prior has three
      // components: a prior for the long run mean of the slope, a prior for the
      // slope's AR coefficient, and a prior for the standard deviation of the
      // AR1 process.
      NEW(GaussianModel, slope_mean_prior)(slope_mean_prior_spec->mu(),
                                           slope_mean_prior_spec->sigma_guess());

      NEW(GaussianModel, slope_ar_prior)(slope_ar1_prior_spec->mu(),
                                         slope_ar1_prior_spec->sigma_guess());
      NEW(ChisqModel, slope_sigma_prior)(slope_prior_spec->prior_df(),
                                         slope_prior_spec->prior_guess());

      // The components have been created.  Time to create the overall prior.
      NEW(NonzeroMeanAr1Sampler, slope_sampler)(slope.get(),
                                                slope_mean_prior,
                                                slope_ar_prior,
                                                slope_sigma_prior);
      // Optional features of the slope prior...
      // Set an upper limit for sigma, if desired.
      if (slope_prior_spec->sigma_upper_limit() > 0) {
        slope_sampler->set_sigma_upper_limit(slope_prior_spec->sigma_upper_limit());
      }

      // Force the slope model to be stationarity, if desired.
      if (slope_ar1_prior_spec->truncate()) {
        slope_sampler->force_stationary();
      }

      if (slope_ar1_prior_spec->positive()) {
        slope_sampler->force_ar1_positive();
      }

      // The slope prior is built and configured.  Pack it in to the trend
      // model.  Note that it goes in the trend model, not the slope model,
      // because it is the trend model's "sample_posterior" method that will be
      // called.
      trend->set_method(slope_sampler);

      // The final task is to set the prior for the initial value of the state.
      trend->set_initial_level_mean(level_prior_spec->initial_value());
      trend->set_initial_slope_mean(slope_prior_spec->initial_value());
      trend->set_initial_level_sd(level_prior_spec->sigma_guess());
      trend->set_initial_slope_sd(slope_prior_spec->sigma_guess());

      if (io_manager()) {
        io_manager()->add_list_element(
            new StandardDeviationListElement(
                level->Sigsq_prm(),
                prefix + "trend.level.sd"));

        io_manager()->add_list_element(
            new UnivariateListElement(
                slope->Mu_prm(),
                prefix + "trend.slope.mean"));
        io_manager()->add_list_element(
            new UnivariateListElement(
                slope->Phi_prm(),
                prefix + "trend.slope.ar.coefficient"));
        io_manager()->add_list_element(
            new StandardDeviationListElement(
                slope->Sigsq_prm(),
                prefix + "trend.slope.sd"));
      }

      return trend;
    }

    namespace {

    //   // This code is shared between the SeasonalStateModel and the
    //   // MonthlyAnnualCycle.
    //   template <class SEASONAL>
    //   void set_initial_state_prior(SEASONAL *component,
    //                                const ScalarStateSpaceSpecification *specification) {
    //     // Set prior distribution for initial state.
    //     SEXP r_initial_state_prior(getListElement(
    //         r_state_component, "initial.state.prior"));
    //     if (Rf_inherits(r_initial_state_prior, "NormalPrior")) {
    //       NormalPrior initial_value_prior_spec(r_initial_state_prior);
    //       component->set_initial_state_variance(
    //           square(initial_value_prior_spec.sigma()));
    //     } else if (Rf_inherits(r_initial_state_prior, "MvnDiagonalPrior")) {
    //       MvnDiagonalPrior initial_value_prior_spec(r_initial_state_prior);
    //       component->set_initial_state_mean(
    //           initial_value_prior_spec.mean());
    //       SpdMatrix variance(initial_value_prior_spec.sd().size());
    //       variance.set_diag(pow(initial_value_prior_spec.sd(), 2));
    //       component->set_initial_state_variance(variance);
    //     } else if (Rf_inherits(r_initial_state_prior, "MvnPrior")) {
    //       MvnPrior spec(r_initial_state_prior);
    //       component->set_initial_state_mean(spec.mu());
    //       component->set_initial_state_variance(spec.Sigma());
    //     }
    //   }

    //   template <class SEASONAL>
    //   void set_posterior_sampler(SEASONAL *component,
    //                              const SdPrior &sigma_prior_spec) {
    //     // Set prior distribution for variance parameter
    //     if (sigma_prior_spec.fixed()) {
    //       Ptr<FixedUnivariateSampler> sampler(
    //           new FixedUnivariateSampler(
    //               component->Sigsq_prm(),
    //               component->sigsq()));
    //       component->set_method(sampler);
    //     } else {
    //       Ptr<ZeroMeanGaussianConjSampler> sampler(
    //           new ZeroMeanGaussianConjSampler(component,
    //                                           sigma_prior_spec.prior_df(),
    //                                           sigma_prior_spec.prior_guess()));

    //       if (sigma_prior_spec.upper_limit() > 0) {
    //         sampler->set_sigma_upper_limit(sigma_prior_spec.upper_limit());
    //       }
    //       component->set_method(sampler);
    //     }
    //   }


      // Returns the position of the specified state model in the state space
      // model.  If the state space model is nullptr or the state model is not
      // found, then -1 is returned.
      int StateModelPosition(StateSpaceModelBase *model,
                             StateModel *state_model) {
        if (!model) {
          return -1;
        } else {
          for (int i = 0; i < model->number_of_state_models(); ++i) {
            if (model->state_model(i).get() == state_model) {
              return i;
            }
          }
        }
        return -1;
      }

    }  // namespace

    // Args:
    //   holiday_spec: An R list describing a single holiday.  Must inherit from
    //     class 'Holiday' and from a specific holiday class.  Class membership
    //     is used to dispatch the right method of object creation.
    // Returns:
    //   A raw pointer to a Holiday object of the type specified by the R-class
    //   attribute of holiday_spec.  The pointer should be immediately caught by
    //   a Ptr.
    // Ptr<Holiday> StateModelFactory::CreateHoliday(const ScalarSateSpaceSpecification *specification, const Date &holiday)
    //   if (Rf_inherits(holiday_spec, "NthWeekdayInMonthHoliday")) {
    //     int week = Rf_asInteger(getListElement(holiday_spec, "week.number"));
    //     std::string day = ToString(getListElement(holiday_spec, "day.of.week"));
    //     std::string month = ToString(getListElement(holiday_spec, "month"));
    //     return new NthWeekdayInMonthHoliday(
    //         week, str2day(day), str2month(month),
    //         Rf_asInteger(getListElement(holiday_spec, "days.before")),
    //         Rf_asInteger(getListElement(holiday_spec, "days.after")));
    //   } else if (Rf_inherits(holiday_spec, "LastWeekdayInMonthHoliday")) {
    //     std::string day = ToString(getListElement(holiday_spec, "day.of.week"));
    //     std::string month = ToString(getListElement(holiday_spec, "month"));
    //     return new LastWeekdayInMonthHoliday(
    //         str2day(day),
    //         str2month(month),
    //         Rf_asInteger(getListElement(holiday_spec, "days.before")),
    //         Rf_asInteger(getListElement(holiday_spec, "days.after")));
    //   } else if (Rf_inherits(holiday_spec, "FixedDateHoliday")) {
    //     int day = Rf_asInteger(getListElement(holiday_spec, "day"));
    //     std::string month = ToString(getListElement(holiday_spec, "month"));
    //     return new FixedDateHoliday(
    //         str2month(month),
    //         day,
    //         Rf_asInteger(getListElement(holiday_spec, "days.before")),
    //         Rf_asInteger(getListElement(holiday_spec, "days.after")));
    //   } else if (Rf_inherits(holiday_spec, "DateRangeHoliday")) {
    //     std::vector<Date> start_date = ToBoomDateVector(getListElement(
    //         holiday_spec, "start.date", true));
    //     std::vector<Date> end_date = ToBoomDateVector(getListElement(
    //         holiday_spec, "end.date", true));
    //     return new DateRangeHoliday(start_date, end_date);
    //   } else if (Rf_inherits(holiday_spec, "NamedHoliday")) {
    //     return BOOM::CreateNamedHoliday(
    //         ToString(getListElement(holiday_spec, "name")),
    //         Rf_asInteger(getListElement(holiday_spec, "days.before")),
    //         Rf_asInteger(getListElement(holiday_spec, "days.after")));
    //   } else {
    //     report_error("Unknown holiday type passed to CreateHoliday.");
    //     return nullptr;
    //   }
    // }

    SeasonalStateModel * StateModelFactory::CreateSeasonal(const ScalarStateSpaceSpecification *specification,
        SeasonSpecification* season, const std::string &prefix) {

      if (!season) {
        report_error("CreateSeasonal: empty season");
      }

      int nseasons = season->number_of_seasons();
      int season_duration = season->duration();
      std::shared_ptr<PriorSpecification> prior_spec = specification->seasonal_sigma_prior();
      if (!prior_spec) {
        report_error("CreateSeasonal: empty seasonal_sigma_prior");
      }

      SeasonalStateModel * seasonal(
          new SeasonalStateModel(nseasons, season_duration));
      seasonal->set_sigsq(square(prior_spec->prior_guess()));

      // Set prior distribution for initial state.
      seasonal->set_initial_state_variance(
          square(prior_spec->sigma_guess()));

      // Set prior distribution for variance parameter
      if (prior_spec->fixed()) {
        Ptr<FixedUnivariateSampler> sampler(
            new FixedUnivariateSampler(
                seasonal->Sigsq_prm(),
                seasonal->sigsq()));
        seasonal->set_method(sampler);
      } else {
        Ptr<ZeroMeanGaussianConjSampler> sampler(
            new ZeroMeanGaussianConjSampler(seasonal,
                                            prior_spec->prior_df(),
                                            prior_spec->prior_guess()));

        if (prior_spec->sigma_upper_limit() > 0) {
          sampler->set_sigma_upper_limit(prior_spec->sigma_upper_limit());
        }
        seasonal->set_method(sampler);
      }

      std::ostringstream parameter_name;
      parameter_name  <<  "sigma.seasonal" << "." << nseasons;
      if (season_duration > 1) parameter_name << "." << season_duration;

      if (io_manager()) {
        io_manager()->add_list_element(new StandardDeviationListElement(
            seasonal->Sigsq_prm(),
            prefix + parameter_name.str()));
      }
      return seasonal;
    }
    // //======================================================================
    // MonthlyAnnualCycle *StateModelFactory::CreateMonthlyAnnualCycle(const ScalarStateSpaceSpecification *specification) {
    //   Date date_of_first_observation(
    //       Rf_asInteger(getListElement(
    //           r_state_component, "first.observation.month")),
    //       Rf_asInteger(getListElement(
    //           r_state_component, "first.observation.day")),
    //       Rf_asInteger(getListElement(
    //           r_state_component, "first.observation.year")));
    //   MonthlyAnnualCycle *monthly =
    //       new MonthlyAnnualCycle(date_of_first_observation);
    //   SdPrior sigma_prior_spec(specification->sigma_prior());
    //   monthly->set_sigsq(square(sigma_prior_spec.initial_value()));
    //   set_initial_state_prior(monthly, specification);
    //   set_posterior_sampler(monthly, sigma_prior_spec);
    //   return monthly;
    // }

    // //======================================================================
    // // Creates a random walk holiday state model.
    // // Args:
    // //   r_state_component: An R object inheriting from class
    // //     "RandomWalkHolidayStateModel".
    // //   prefix: An optional prefix to be prepended to the name of the state
    // //     component in the io_manager.
    // RandomWalkHolidayStateModel *
    // StateModelFactory::CreateRandomWalkHolidayStateModel(const ScalarStateSpaceSpecification *specification) {

    //   SEXP r_holiday = getListElement(r_state_component, "holiday");
    //   Ptr<Holiday> holiday = CreateHoliday(r_holiday);
    //   std::string holiday_name = ToString(getListElement(r_holiday, "name"));
          
    //   Date time0 = ToBoomDate(getListElement(r_state_component, "time0"));
    //   SdPrior sigma_prior_spec(getListElement(
    //       r_state_component, "sigma.prior"));
    //   NormalPrior initial_value_prior_spec(getListElement(
    //       r_state_component, "initial.state.prior"));

    //   RandomWalkHolidayStateModel * holiday_model
    //       = new RandomWalkHolidayStateModel(holiday, time0);
    //   holiday_model->set_sigsq(square(sigma_prior_spec.initial_value()));

    //   //------------------------------------------------------------
    //   // Set prior distribution for initial state
    //   Vector initial_state_mean(holiday_model->state_dimension(), 0.0);
    //   SpdMatrix initial_state_variance(holiday_model->state_dimension());
    //   initial_state_variance.set_diag(square(initial_value_prior_spec.sigma()));
    //   holiday_model->set_initial_state_mean(initial_state_mean);
    //   holiday_model->set_initial_state_variance(initial_state_variance);

    //   //------------------------------------------------------------
    //   // Set prior distribution for innovation variance parameter
    //   if (sigma_prior_spec.fixed()) {
    //     Ptr<FixedUnivariateSampler> sampler(
    //         new FixedUnivariateSampler(
    //             holiday_model->Sigsq_prm(),
    //             holiday_model->sigsq()));
    //     holiday_model->set_method(sampler);
    //   } else {
    //     Ptr<ZeroMeanGaussianConjSampler> sampler(
    //         new ZeroMeanGaussianConjSampler(
    //             holiday_model,
    //             sigma_prior_spec.prior_df(),
    //             sigma_prior_spec.prior_guess()));
    //     holiday_model->set_method(sampler);
    //   }

    //   std::ostringstream parameter_name;
    //   parameter_name  <<  "sigma." << holiday_name;
    //   // Add information about this parameter to the io_manager
    //   if (io_manager()) {
    //     io_manager()->add_list_element(new StandardDeviationListElement(
    //         holiday_model->Sigsq_prm(),
    //         prefix + parameter_name.str()));
    //   }
    //   return holiday_model;
    // }

    // //=========================================================================
    // ScalarRegressionHolidayStateModel *
    // StateModelFactory::CreateRegressionHolidayStateModel(
    //  ,
    //     ScalarStateSpaceModelBase *model) {
    //   Date time_zero = ToBoomDate(getListElement(
    //       r_state_specification, "time0"));
    //   NormalPrior prior_spec(getListElement(r_state_specification, "prior"));
    //   NEW(GaussianModel, prior)(prior_spec.mu(), prior_spec.sigsq());
    //   ScalarRegressionHolidayStateModel *holiday_model =
    //       new ScalarRegressionHolidayStateModel(time_zero, model, prior);
    //   ImbueRegressionHolidayStateModel(holiday_model, r_state_specification, prefix);
    //   return holiday_model;
    // }

    // void StateModelFactory::ImbueRegressionHolidayStateModel(
    //     RegressionHolidayStateModel *holiday_model,
    //     const ScalarStateSpaceSpecification *specification) {
    //   RMemoryProtector holiday_list_protector;
    //   SEXP r_holidays = holiday_list_protector.protect(
    //       getListElement(r_state_specification, "holidays"));

    //   int number_of_holidays = Rf_length(r_holidays);
    //   for (int i = 0; i < number_of_holidays; ++i) {
    //     RMemoryProtector holiday_protector;
    //     SEXP r_holiday = holiday_protector.protect(VECTOR_ELT(r_holidays, i));
    //     Ptr<Holiday> holiday = CreateHoliday(r_holiday);
    //     std::string holiday_name =
    //         prefix + ToString(getListElement(r_holiday, "name"));
    //     holiday_model->add_holiday(holiday);
    //     io_manager()->add_list_element(new VectorListElement(
    //         holiday_model->holiday_pattern_parameter(i),
    //         holiday_name));
    //   }
    // }

    // //=========================================================================
    // DynamicInterceptRegressionHolidayStateModel *
    // StateModelFactory::CreateDynamicInterceptRegressionHolidayStateModel(
    //  ,
    //     DynamicInterceptRegressionModel *model) {
    //   Date time_zero = ToBoomDate(getListElement(
    //       r_state_specification, "time0"));
    //   NormalPrior prior_spec(getListElement(r_state_specification, "prior"));
    //   NEW(GaussianModel, prior)(prior_spec.mu(), prior_spec.sigsq());
    //   DynamicInterceptRegressionHolidayStateModel *holiday_model =
    //       new DynamicInterceptRegressionHolidayStateModel(
    //           time_zero, model, prior);
    //   ImbueRegressionHolidayStateModel(
    //       holiday_model, r_state_specification, prefix);
    //   return holiday_model;
    // }

    // //=========================================================================
    // // Args:
    // //   r_state_specification: An R object of class
    // //     ShrinkageRegressionHolidayModel detailing the model to be built.
    // ScalarHierarchicalRegressionHolidayStateModel *
    // StateModelFactory::CreateHierarchicalRegressionHolidayStateModel(
    //     const ScalarStateSpaceSpecification *specification,
    //     ScalarStateSpaceModelBase *model) {
    //   Date time_zero = ToBoomDate(getListElement(
    //       r_state_specification, "time0"));
    //   ScalarHierarchicalRegressionHolidayStateModel *holiday_model =
    //       new ScalarHierarchicalRegressionHolidayStateModel(time_zero, model);
    //   ImbueHierarchicalRegressionHolidayStateModel(
    //       holiday_model, r_state_specification, prefix);
    //   return holiday_model;
    // }
      
    // void StateModelFactory::ImbueHierarchicalRegressionHolidayStateModel(
    //     HierarchicalRegressionHolidayStateModel *holiday_model,
    //     const ScalarStateSpaceSpecification *specification) {
    //   SEXP r_holidays = getListElement(r_state_specification, "holidays");
    //   int number_of_holidays = Rf_length(r_holidays);
    //   std::vector<std::string> holiday_names;
    //   for (int i = 0; i < number_of_holidays; ++i) {
    //     SEXP r_holiday = VECTOR_ELT(r_holidays, i);
    //     Ptr<Holiday> holiday = CreateHoliday(r_holiday);
    //     holiday_names.push_back(ToString(getListElement(
    //         r_holiday, "name")));
    //     holiday_model->add_holiday(holiday);
    //   }

    //   // Unpack the priors and set the posterior sampler.
    //   MvnPrior coefficient_mean_prior_spec(
    //       getListElement(r_state_specification, "coefficient.mean.prior"));
    //   NEW(MvnModel, coefficient_mean_prior)(
    //       coefficient_mean_prior_spec.mu(),
    //       coefficient_mean_prior_spec.Sigma());
    //   InverseWishartPrior coefficient_variance_prior_spec(
    //       getListElement(r_state_specification, "coefficient.variance.prior"));
    //   NEW(WishartModel, coefficient_variance_prior)(
    //       coefficient_variance_prior_spec.variance_guess_weight(),
    //       coefficient_variance_prior_spec.variance_guess());

    //   NEW(HierGaussianRegressionAsisSampler, sampler)(
    //       holiday_model->model(),
    //       coefficient_mean_prior,
    //       coefficient_variance_prior,
    //       nullptr);
    //   // This is one of the rare situations where the model holding the sampler
    //   // is not the model being updated.  holiday_model owns the model being
    //   // updated.
    //   holiday_model->set_method(sampler);

    //   // Set up the io_manager
    //   std::vector<Ptr<VectorParams>> holiday_coefficients;
    //   for (int i = 0; i < number_of_holidays; ++i) {
    //     holiday_coefficients.push_back(
    //         holiday_model->model()->data_model(i)->coef_prm());
    //   }
    //   HierarchicalVectorListElement *coefficient_io =  
    //       new HierarchicalVectorListElement(
    //           holiday_coefficients,
    //           prefix + "holiday.coefficients");
    //   coefficient_io->set_group_names(holiday_names);
    //   io_manager()->add_list_element(coefficient_io);

    //   io_manager()->add_list_element(
    //       new VectorListElement(holiday_model->model()->prior()->Mu_prm(),
    //                             prefix + "holiday.coefficient.mean"));
    //   io_manager()->add_list_element(
    //       new SpdListElement(holiday_model->model()->prior()->Sigma_prm(),
    //                          prefix + "holiday.coefficient.variance"));
    // }

    // DynamicInterceptHierarchicalRegressionHolidayStateModel *
    // StateModelFactory::CreateDIHRHSM(const ScalarStateSpaceSpecification *specification,
    //                                  DynamicInterceptRegressionModel *model) {
    //   Date time_zero = ToBoomDate(getListElement(
    //       r_state_specification, "time0"));
    //   DynamicInterceptHierarchicalRegressionHolidayStateModel *holiday_model =
    //       new DynamicInterceptHierarchicalRegressionHolidayStateModel(
    //           time_zero, model);
    //   ImbueHierarchicalRegressionHolidayStateModel(
    //       holiday_model, r_state_specification, prefix);
    //   return holiday_model;
    // }
    
    //======================================================================
    ArStateModel * StateModelFactory::CreateArStateModel(
        const ScalarStateSpaceSpecification *specification,
        const std::string &prefix) {
      std::shared_ptr<PriorSpecification> sigma = specification->sigma_prior();
      if (!sigma) {
        report_error("CreateArStateModel: empty sigma_prior");
      }
      int number_of_lags = specification->ar_order();
      ArStateModel *state_model(new ArStateModel(number_of_lags));

      NEW(ChisqModel, siginv_prior)(sigma->prior_df(),
                                    sigma->prior_guess());

      NEW(ArPosteriorSampler, sampler)(state_model,
                                       siginv_prior);
      if(sigma->sigma_upper_limit() > 0) {
        sampler->set_sigma_upper_limit(sigma->sigma_upper_limit());
      }
      state_model->set_method(sampler);

      if (io_manager()) {
        std::ostringstream phi_parameter_name;
        phi_parameter_name << prefix << "AR" << number_of_lags
                           << ".coefficients";
        io_manager()->add_list_element(new GlmCoefsListElement(
            state_model->Phi_prm(),
            phi_parameter_name.str()));

        std::ostringstream sigma_parameter_name;
        sigma_parameter_name << prefix << "AR" << number_of_lags << ".sigma";
        io_manager()->add_list_element(new StandardDeviationListElement(
            state_model->Sigsq_prm(),
            sigma_parameter_name.str()));
      }
      return state_model;
    }
    //======================================================================
    ArStateModel * StateModelFactory::CreateAutoArStateModel(
        const ScalarStateSpaceSpecification *specification,
        const std::string &prefix) {
      int number_of_lags = specification->ar_order();
      ArStateModel *state_model(new ArStateModel(number_of_lags));
      std::shared_ptr<PriorSpecification> ar_prior = specification->ar_prior();
      if (!ar_prior) {
        report_error("CreateAutoArStateModel: empty ar_prior");
      }
      ArSpikeSlabPrior prior_spec(ar_prior->prior_inclusion_probabilities(),
          ar_prior->prior_mean(), ar_prior->prior_precision(), ar_prior->max_flips(),
          ar_prior->prior_df(), ar_prior->prior_guess(), ar_prior->sigma_upper_limit(),
          ar_prior->truncate());
      NEW(ArSpikeSlabSampler, sampler)(state_model,
                                       prior_spec.slab(),
                                       prior_spec.spike(),
                                       prior_spec.siginv_prior(),
                                       prior_spec.truncate());
      if (prior_spec.max_flips() > 0) {
        sampler->limit_model_selection(prior_spec.max_flips());
      }

      if (prior_spec.sigma_upper_limit() > 0) {
        sampler->set_sigma_upper_limit(prior_spec.sigma_upper_limit());
      }

      state_model->set_method(sampler);

      if (io_manager()) {
        std::ostringstream phi_parameter_name;
        phi_parameter_name << prefix << "AR" << number_of_lags
                           << ".coefficients";
        std::vector<std::string> column_names;
        for (int i = 0; i < number_of_lags; ++i) {
          ostringstream column_name;
          column_name << "lag." << i + 1;
          column_names.push_back(column_name.str());
        }
        io_manager()->add_list_element(new GlmCoefsListElement(
            state_model->Phi_prm(),
            phi_parameter_name.str(),
            column_names));

        std::ostringstream sigma_parameter_name;
        sigma_parameter_name << prefix << "AR" << number_of_lags << ".sigma";
        io_manager()->add_list_element(new StandardDeviationListElement(
            state_model->Sigsq_prm(),
            sigma_parameter_name.str()));
      }
      return state_model;
    }

    //======================================================================
    // This is a callback designed to be used with a NativeMatrixListElement in
    // an Python io_manager.  Invoking this callback grabs the dynamic regression
    // component of the model's current state, allowing the model to store the
    // dynamic regression coefficients at each iteration.
    class DynamicRegressionRandomWalkStateCallback : public BOOM::MatrixIoCallback {
     public:
      DynamicRegressionRandomWalkStateCallback(
          BOOM::StateSpaceModelBase *model,
          DynamicRegressionStateModel *state_model)
          : model_(model),
            state_model_(state_model),
            model_position_(-1) {}

      // There is one row for each dynamic regression coefficient.
      int nrow() const override {return state_model_->state_dimension(); }
      int ncol() const override {return model_->time_dimension();}
      BOOM::Matrix get_matrix() const override {
        if (model_position_ < 0) determine_model_position();
        return model_->full_state_subcomponent(model_position_).to_matrix();
      }

      void determine_model_position() const {
        model_position_ = StateModelPosition(model_, state_model_);
      }

     private:
      BOOM::StateSpaceModelBase *model_;
      DynamicRegressionStateModel *state_model_;
      mutable int model_position_;
    };

    // When called, this callback will record the current values of the
    // coefficients in a DynamicRegressionArStateModel.
    class DynamicRegressionArStateCallback : public BOOM::MatrixIoCallback {
     public:
      DynamicRegressionArStateCallback(BOOM::StateSpaceModelBase *model,
                                       DynamicRegressionArStateModel *state_model)
          : model_(model),
            state_model_(state_model),
            model_position_(-1) {}
      int nrow() const override {return state_model_->xdim();}
      int ncol() const override {return model_->time_dimension();}
      BOOM::Matrix get_matrix() const override {
        if (model_position_ < 0) DetermineModelPosition();
        int lags = state_model_->number_of_lags();
        Matrix ans(nrow(), ncol());
        int row_index = 0;
        const ConstSubMatrix
            state(model_->full_state_subcomponent(model_position_));
        for (int i = 0; i < nrow(); ++i) {
          ans.row(i) = state.row(row_index);
          row_index += lags;
        }
        return ans;
      }

      void DetermineModelPosition() const {
        model_position_ = StateModelPosition(model_, state_model_);
      }

     private:
      BOOM::StateSpaceModelBase *model_;
      DynamicRegressionArStateModel *state_model_;
      mutable int model_position_;
    };

    //======================================================================
    void SetIndependentDynamicRegressionModelPriors(
        DynamicRegressionStateModel *model,
        const ScalarStateSpaceSpecification * specification) {
      std::shared_ptr<PriorSpecification> sigma_prior = specification->sigma_prior();
      std::vector<Ptr<GammaModelBase>> precision_priors;
      Vector sigma_max(model->xdim());
      for (int i = 0; i < model->xdim(); ++i) {
        precision_priors.push_back(new ChisqModel(
            sigma_prior->prior_df(),
            sigma_prior->prior_guess()));
        sigma_max[i] = sigma_prior->sigma_upper_limit();
      }
      NEW(DynamicRegressionIndependentPosteriorSampler, sampler)(
          model, precision_priors);
      for (int i = 0; i < model->xdim(); ++i) {
        if (sigma_max[i] > 0 || std::isfinite(sigma_max[i])) {
          for (int i = 0; i < model->xdim(); ++i) {
            sampler->set_sigma_max(i, sigma_max[i]);
          }
        }
      }
      model->set_method(sampler);      
    }
    //======================================================================
    void SetHierarchicalDynamicRegressionModelPrior(
        DynamicRegressionStateModel *model,
        const ScalarStateSpaceSpecification *specification,
        std::shared_ptr<PythonListIoManager> io_manager,
        const std::string &prefix) {
      Ptr<DoubleModel> sigma_mean_prior =
          create_double_model(specification->hierarchical_regression_specification()->sigma_mean_prior());
      Ptr<DoubleModel> shrinkage_parameter_prior =
          create_double_model(specification->hierarchical_regression_specification()->shrinkage_prior());

      NEW(GammaModel, siginv_prior)(1, 1);
      NEW(GammaPosteriorSampler, hyperparameter_sampler)(
          siginv_prior.get(),
          sigma_mean_prior,
          shrinkage_parameter_prior);
      siginv_prior->set_method(hyperparameter_sampler);

      NEW(DynamicRegressionPosteriorSampler, sampler)(
          model, siginv_prior);
      double sigma_max = specification->sigma_prior()->sigma_upper_limit();
      if (std::isfinite(sigma_max)) {
        sampler->set_sigma_max(sigma_max);
      }
      model->set_method(sampler);

      if (io_manager) {
        // Store the hyperparameters describing the model for 1.0 / sigma^2.
        io_manager->add_list_element(new UnivariateListElement(
            siginv_prior->Alpha_prm(),
            prefix + "siginv_shape_hyperparameter"));

        io_manager->add_list_element(new UnivariateListElement(
            siginv_prior->Beta_prm(),
            prefix + "siginv_scale_hyperparameter"));
      }      
    }
    //======================================================================
    void SetDynamicRegressionModelPrior(
        DynamicRegressionStateModel *model,
        const ScalarStateSpaceSpecification *specification,
        std::shared_ptr<PythonListIoManager> io_manager,
        const std::string &prefix) {
      if (specification->hierarchical_regression_specification()) {
        SetHierarchicalDynamicRegressionModelPrior(model, specification, io_manager, prefix);
      } else {
        SetIndependentDynamicRegressionModelPriors(model, specification);
      }
    }

    DynamicRegressionStateModel *
    StateModelFactory::CreateDynamicRegressionStateModel(
        const ScalarStateSpaceSpecification *specification,
        StateSpaceRegressionManagedModel *model,
        const std::string &prefix) {
      if (!model) {
        report_error("CreateDynamicRegressionStateModel: empty model");
      }
      if (!(model->sampling_model())) {
        report_error("CreateDynamicRegressionStateModel: empty sampling_model");
      }

      IdentifyDynamicRegression(model->sampling_model()->number_of_state_models());
      Matrix predictors = model->predictors();
      if (predictors.ncol()==0) {
        report_error("CreateDynamicRegressionStateModel: empty predictors");
      }

      std::vector<std::string> xnames = specification->predictor_names();
      if (xnames.empty()) {
        xnames.reserve(ncol(predictors));
        for (int i = 0; i < ncol(predictors); ++i) {
          std::ostringstream name_maker;
          name_maker << "V" << i+1;
          xnames.push_back(name_maker.str());
        }
      }

      DynamicRegressionStateModel * dynamic_regression(
          new DynamicRegressionStateModel(predictors));
      dynamic_regression->set_xnames(xnames);
      SetDynamicRegressionModelPrior(dynamic_regression, specification, io_manager(), prefix);

      if (io_manager()) {
        // Store the standard deviations for each variable.
        for (int i = 0; i < ncol(predictors); ++i) {
          std::ostringstream vname;
          vname << prefix << xnames[i] << ".sigma";
          io_manager()->add_list_element(new StandardDeviationListElement(
              dynamic_regression->Sigsq_prm(i),
              vname.str()));
        }

        NativeMatrixListElement *dynamic_regression_coefficients(
            new NativeMatrixListElement(
                new DynamicRegressionRandomWalkStateCallback(
                    model->sampling_model(),
                    dynamic_regression),
                "dynamic.regression.coefficients",
                nullptr));
        dynamic_regression_coefficients->set_row_names(xnames);
        AddPostStateListElement(dynamic_regression_coefficients);
      }

      return dynamic_regression;
    }

    //======================================================================
    DynamicRegressionArStateModel *
    StateModelFactory::CreateDynamicRegressionArStateModel(
        const ScalarStateSpaceSpecification *specification,
        StateSpaceRegressionManagedModel *model,
        const std::string &prefix) {
      if (!model) {
        report_error("CreateDynamicRegressionArStateModel: empty model");
      }
      if (!model->sampling_model()) {
        report_error("CreateDynamicRegressionArStateModel: empty sampling_model");
      }
      IdentifyDynamicRegression(model->sampling_model()->number_of_state_models());

      Matrix predictors = model->predictors();
      if (predictors.ncol()==0) {
        report_error("CreateDynamicRegressionArStateModel: empty predictors");
      }
      std::vector<std::string> xnames =
          specification->predictor_names();
      if (xnames.empty()) {
        xnames.reserve(ncol(predictors));
        for (int i = 0; i < ncol(predictors); ++i) {
          std::ostringstream name_maker;
          name_maker << "V" << i+1;
          xnames.push_back(name_maker.str());
        }
      }

      // Build the model.
      int lags = specification->ar_order();
      DynamicRegressionArStateModel * dynamic_regression(
          new DynamicRegressionArStateModel(predictors, lags));
      dynamic_regression->set_xnames(xnames);

      // Set the prior and the posterior sampler.
      std::shared_ptr<PriorSpecification> sigma_prior = specification->sigma_prior();
      if (!sigma_prior) {
        report_error("CreateDynamicRegressionArStateModel: empty sigma_prior");
      }
      std::vector<Ptr<GammaModelBase>> siginv_priors;
      siginv_priors.reserve(ncol(predictors));
      for (int i = 0; i < ncol(predictors); ++i) {
        NEW(ChisqModel, siginv_prior)(sigma_prior->prior_df(),
                                      sigma_prior->prior_guess());
        siginv_priors.push_back(siginv_prior);
      }

      NEW(DynamicRegressionArPosteriorSampler, sampler)(
          dynamic_regression, siginv_priors);
      dynamic_regression->set_method(sampler);


      if (io_manager()) {
        std::vector<std::string> lag_names;
        lag_names.reserve(lags);
        for (int i = 0; i < lags; ++i) {
          std::ostringstream lag_name;
          lag_name << "lag." << i+1;
          lag_names.push_back(lag_name.str());
        }

        for (int i = 0; i < ncol(predictors); ++i) {
          std::ostringstream sigsq_param_name;
          sigsq_param_name << prefix << xnames[i] << ".sigma";
          io_manager()->add_list_element(new StandardDeviationListElement(
              dynamic_regression->coefficient_model(i)->Sigsq_prm(),
              sigsq_param_name.str()));

          std::ostringstream coefficient_param_name;
          coefficient_param_name << prefix << xnames[i] << ".ar.coefficients";
          io_manager()->add_list_element(new GlmCoefsListElement(
              dynamic_regression->coefficient_model(i)->coef_prm(),
              coefficient_param_name.str(),
              lag_names));
        }

        // Create a state component for recording the individual dynamic
        // regression coefficients, but install it after all the components that
        // track the parameters of the state models.
        NativeMatrixListElement *dynamic_regression_coefficients(
            new NativeMatrixListElement(
                new DynamicRegressionArStateCallback(model->sampling_model(), dynamic_regression),
                "dynamic.regression.coefficients",
                nullptr));
        dynamic_regression_coefficients->set_row_names(xnames);
        AddPostStateListElement(dynamic_regression_coefficients);
      }
      return dynamic_regression;
    }

  }  // namespace pybsts
}  // namespace BOOM
