# Part of the rstanarm package for estimating model parameters
# Copyright (C) 2013, 2014, 2015, 2016, 2017 Trustees of Columbia University
# Copyright (C) 2016, 2017 Sam Brilleman
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#' Bayesian joint longitudinal and time-to-event models via Stan
#' 
#' \if{html}{\figure{stanlogo.png}{options: width="25" alt="https://mc-stan.org/about/logo/"}}
#' Fits a shared parameter joint model for longitudinal and time-to-event 
#' (e.g. survival) data under a Bayesian framework using Stan.
#' 
#' @export
#' @template args-dots
#' @template args-prior_PD
#' @template args-algorithm
#' @template args-adapt_delta
#' @template args-max_treedepth
#' @template args-QR
#' @template args-sparse
#' 
#' @param formulaLong A two-sided linear formula object describing both the 
#'   fixed-effects and random-effects parts of the longitudinal submodel,
#'   similar in vein to formula specification in the \strong{lme4} package
#'   (see \code{\link[lme4]{glmer}} or the \strong{lme4} vignette for details). 
#'   Note however that the double bar (\code{||}) notation is not allowed 
#'   when specifying the random-effects parts of the formula, and neither
#'   are nested grouping factors (e.g. \code{(1 | g1/g2))} or 
#'   \code{(1 | g1:g2)}, where \code{g1}, \code{g2} are grouping factors. 
#'   Offset terms can also be included in the model formula.
#'   For a multivariate joint model (i.e. more than one longitudinal marker) 
#'   this should be a list of such formula objects, with each element
#'   of the list providing the formula for one of the longitudinal submodels.
#' @param dataLong A data frame containing the variables specified in
#'   \code{formulaLong}. If fitting a multivariate joint model, then this can
#'   be either a single data frame which contains the data for all 
#'   longitudinal submodels, or it can be a list of data frames where each
#'   element of the list provides the data for one of the longitudinal 
#'   submodels.
#' @param formulaEvent A two-sided formula object describing the event
#'   submodel. The left hand side of the formula should be a \code{Surv()} 
#'   object. See \code{\link[survival]{Surv}}.
#' @param dataEvent A data frame containing the variables specified in
#'   \code{formulaEvent}.
#' @param time_var A character string specifying the name of the variable 
#'   in \code{dataLong} which represents time.
#' @param id_var A character string specifying the name of the variable in
#'   \code{dataLong} which distinguishes between individuals. This can be
#'   left unspecified if there is only one grouping factor (which is assumed
#'   to be the individual). If there is more than one grouping factor (i.e.
#'   clustering beyond the level of the individual) then the \code{id_var}
#'   argument must be specified.
#' @param family The family (and possibly also the link function) for the 
#'   longitudinal submodel(s). See \code{\link[lme4]{glmer}} for details. 
#'   If fitting a multivariate joint model, then this can optionally be a
#'   list of families, in which case each element of the list specifies the
#'   family for one of the longitudinal submodels.
#' @param assoc A character string or character vector specifying the joint
#'   model association structure. Possible association structures that can
#'   be used include: "etavalue" (the default); "etaslope"; "etaauc"; 
#'   "muvalue"; "muslope"; "muauc"; "shared_b"; "shared_coef"; or "null". 
#'   These are described in the \strong{Details} section below. For a multivariate 
#'   joint model, different association structures can optionally be used for 
#'   each longitudinal submodel by specifying a list of character
#'   vectors, with each element of the list specifying the desired association 
#'   structure for one of the longitudinal submodels. Specifying \code{assoc = NULL}
#'   will fit a joint model with no association structure (equivalent  
#'   to fitting separate longitudinal and time-to-event models). It is also 
#'   possible to include interaction terms between the association term 
#'   ("etavalue", "etaslope", "muvalue", "muslope") and observed data/covariates. 
#'   It is also possible, when fitting a multivariate joint model, to include 
#'   interaction terms between the association terms ("etavalue" or "muvalue") 
#'   corresponding to the different longitudinal outcomes. See the 
#'   \strong{Details} section as well as the \strong{Examples} below.
#' @param lag_assoc A non-negative scalar specifying the time lag that should be
#'   used for the association structure. That is, the hazard of the event at 
#'   time \emph{t} will be assumed to be associated with the value/slope/auc of 
#'   the longitudinal marker at time \emph{t-u}, where \emph{u} is the time lag.
#'   If fitting a multivariate joint model, then a different time lag can be used
#'   for each longitudinal marker by providing a numeric vector of lags, otherwise
#'   if a scalar is provided then the specified time lag will be used for all 
#'   longitudinal markers. Note however that only one time lag  can be specified 
#'   for linking each longitudinal marker to the 
#'   event, and that that time lag will be used for all association structure
#'   types (e.g. \code{"etavalue"}, \code{"etaslope"}, \code{"etaauc"}, 
#'   \code{"muvalue"}, etc) that are specified for that longitudinal marker in
#'   the \code{assoc} argument.
#' @param grp_assoc Character string specifying the method for combining information
#'   across lower level units clustered within an individual when forming the
#'   association structure. This is only relevant when a grouping factor is  
#'   specified in \code{formulaLong} that corresponds to clustering within 
#'   individuals. This can be specified as either \code{"sum"}, \code{mean},
#'   \code{"min"} or \code{"max"}. For example, specifying \code{grp_assoc = "sum"}
#'   indicates that the association structure should be based on a summation across 
#'   the lower level units clustered within an individual, or specifying
#'   \code{grp_assoc = "mean"}  indicates that the association structure 
#'   should be based on the mean (i.e. average) taken across the lower level 
#'   units clustered within an individual.
#'   So, for example, specifying \code{assoc = "muvalue"} 
#'   and \code{grp_assoc = "sum"} would mean that the log hazard at time 
#'   \emph{t} for individual \emph{i} would be linearly related to the sum of
#'   the expected values at time \emph{t} for each of the lower level 
#'   units (which may be for example tumor lesions) clustered within that 
#'   individual. 
#' @param scale_assoc A non-zero numeric value specifying an optional scaling 
#'   parameter for the association structure. This multiplicatively scales the 
#'   value/slope/auc of the longitudinal marker by \code{scale_assoc} within the 
#'   event submodel. When fitting a multivariate joint model, a scaling parameter 
#'   must be specified for each longitudinal submodel using a vector of numeric 
#'   values. Note that only one scaling parameter can be specified for each 
#'   longitudinal submodel, and it will be used for all association structure 
#'   types (e.g. \code{"etavalue"}, \code{"etaslope"}, \code{"etaauc"}, 
#'   \code{"muvalue"}, etc) that are specified for that longitudinal marker in
#'   the \code{assoc} argument.
#' @param basehaz A character string indicating which baseline hazard to use
#'   for the event submodel. Options are a B-splines approximation estimated 
#'   for the log baseline hazard (\code{"bs"}, the default), a Weibull 
#'   baseline hazard (\code{"weibull"}), or a piecewise
#'   constant baseline hazard (\code{"piecewise"}). (Note however that there  
#'   is currently limited post-estimation functionality available for
#'   models estimated using a piecewise constant baseline hazard).
#' @param basehaz_ops A named list specifying options related to the baseline
#'   hazard. Currently this can include: \cr
#'   \describe{
#'     \item{\code{df}}{A positive integer specifying the degrees of freedom 
#'     for the B-splines if \code{basehaz = "bs"}, or the number of
#'     intervals used for the piecewise constant baseline hazard if 
#'     \code{basehaz = "piecewise"}. The default is 6.}
#'     \item{\code{knots}}{An optional numeric vector specifying the internal knot 
#'     locations for the B-splines if \code{basehaz = "bs"}, or the 
#'     internal cut-points for defining intervals of the piecewise constant 
#'     baseline hazard if \code{basehaz = "piecewise"}. Knots cannot be
#'     specified if \code{df} is specified. If not specified, then the 
#'     default is to use \code{df - 4} knots if \code{basehaz = "bs"},
#'     or \code{df - 1} knots if \code{basehaz = "piecewise"}, which are
#'     placed at equally spaced percentiles of the distribution of
#'     observed event times.}
#'   }
#' @param epsilon The half-width of the central difference used to numerically
#'   calculate the derivate when the \code{"etaslope"} association structure 
#'   is used.   
#' @param qnodes The number of nodes to use for the Gauss-Kronrod quadrature
#'   that is used to evaluate the cumulative hazard in the likelihood function. 
#'   Options are 15 (the default), 11 or 7.
#' @param weights Experimental and should be used with caution. The 
#'   user can optionally supply a 2-column data frame containing a set of
#'   'prior weights' to be used in the estimation process. The data frame should
#'   contain two columns: the first containing the IDs for each individual, and 
#'   the second containing the corresponding weights. The data frame should only
#'   have one row for each individual; that is, weights should be constant 
#'   within individuals.
#' @param init The method for generating the initial values for the MCMC.
#'   The default is \code{"prefit"}, which uses those obtained from 
#'   fitting separate longitudinal and time-to-event models prior to 
#'   fitting the joint model. The separate longitudinal model is a 
#'   (possibly multivariate) generalised linear mixed 
#'   model estimated using variational bayes. This is achieved via the 
#'   \code{\link{stan_mvmer}} function with \code{algorithm = "meanfield"}.
#'   The separate Cox model is estimated using \code{\link[survival]{coxph}}. 
#'   This is achieved
#'   using the and time-to-event models prior  
#'   to fitting the joint model. The separate models are estimated using the
#'   \code{\link[lme4]{glmer}} and \code{\link[survival]{coxph}} functions.
#'   This should provide reasonable initial values which should aid the 
#'   MCMC sampler. Parameters that cannot be obtained from 
#'   fitting separate longitudinal and time-to-event models are initialised 
#'   using the "random" method for \code{\link[rstan]{stan}}.
#'   However it is recommended that any final analysis should ideally
#'   be performed with several MCMC chains each initiated from a different
#'   set of initial values; this can be obtained by setting
#'   \code{init = "random"}. In addition, other possibilities for specifying 
#'   \code{init} are the same as those described for \code{\link[rstan]{stan}}.  
#' @param priorLong,priorEvent,priorEvent_assoc The prior distributions for the 
#'   regression coefficients in the longitudinal submodel(s), event submodel,
#'   and the association parameter(s). Can be a call to one of the various functions 
#'   provided by \pkg{rstanarm} for specifying priors. The subset of these functions 
#'   that can be used for the prior on the coefficients can be grouped into several 
#'   "families":
#'   
#'   \tabular{ll}{
#'     \strong{Family} \tab \strong{Functions} \cr 
#'     \emph{Student t family} \tab \code{normal}, \code{student_t}, \code{cauchy} \cr 
#'     \emph{Hierarchical shrinkage family} \tab \code{hs}, \code{hs_plus} \cr 
#'     \emph{Laplace family} \tab \code{laplace}, \code{lasso} \cr
#'   }
#'   
#'   See the \link[=priors]{priors help page} for details on the families and 
#'   how to specify the arguments for all of the functions in the table above.
#'   To omit a prior ---i.e., to use a flat (improper) uniform prior---
#'   \code{prior} can be set to \code{NULL}, although this is rarely a good
#'   idea.
#'   
#'   \strong{Note:} Unless \code{QR=TRUE}, if \code{prior} is from the Student t
#'   family or Laplace family, and if the \code{autoscale} argument to the 
#'   function used to specify the prior (e.g. \code{\link{normal}}) is left at 
#'   its default and recommended value of \code{TRUE}, then the default or 
#'   user-specified prior scale(s) may be adjusted internally based on the scales
#'   of the predictors. See the \link[=priors]{priors help page} for details on
#'   the rescaling and the \code{\link{prior_summary}} function for a summary of
#'   the priors used for a particular model.
#' @param priorLong_intercept,priorEvent_intercept The prior distributions  
#'   for the intercepts in the longitudinal submodel(s) and event submodel. 
#'   Can be a call to \code{normal}, \code{student_t} or 
#'   \code{cauchy}. See the \link[=priors]{priors help page} for details on 
#'   these functions. To omit a prior on the intercept ---i.e., to use a flat
#'   (improper) uniform prior--- \code{prior_intercept} can be set to
#'   \code{NULL}.
#'   
#'   \strong{Note:} The prior distribution for the intercept is set so it
#'   applies to the value when all predictors are centered. Moreover, 
#'   note that a prior is only placed on the intercept for the event submodel
#'   when a Weibull baseline hazard has been specified. For the B-splines and
#'   piecewise constant baseline hazards there is not intercept parameter that
#'   is given a prior distribution; an intercept parameter will be shown in 
#'   the output for the fitted model, but this just corresponds to the 
#'   necessary post-estimation adjustment in the linear predictor due to the
#'   centering of the predictiors in the event submodel.
#'   
#' @param priorLong_aux The prior distribution for the "auxiliary" parameters
#'   in the longitudinal submodels (if applicable). 
#'   The "auxiliary" parameter refers to a different parameter 
#'   depending on the \code{family}. For Gaussian models \code{priorLong_aux} 
#'   controls \code{"sigma"}, the error 
#'   standard deviation. For negative binomial models \code{priorLong_aux} controls 
#'   \code{"reciprocal_dispersion"}, which is similar to the 
#'   \code{"size"} parameter of \code{\link[stats:NegBinomial]{rnbinom}}:
#'   smaller values of \code{"reciprocal_dispersion"} correspond to 
#'   greater dispersion. For gamma models \code{priorLong_aux} sets the prior on 
#'   to the \code{"shape"} parameter (see e.g., 
#'   \code{\link[stats:GammaDist]{rgamma}}), and for inverse-Gaussian models it is the 
#'   so-called \code{"lambda"} parameter (which is essentially the reciprocal of
#'   a scale parameter). Binomial and Poisson models do not have auxiliary 
#'   parameters. 
#'   
#'   \code{priorLong_aux} can be a call to \code{exponential} to 
#'   use an exponential distribution, or \code{normal}, \code{student_t} or 
#'   \code{cauchy}, which results in a half-normal, half-t, or half-Cauchy 
#'   prior. See \code{\link{priors}} for details on these functions. To omit a 
#'   prior ---i.e., to use a flat (improper) uniform prior--- set 
#'   \code{priorLong_aux} to \code{NULL}.
#'   
#'   If fitting a multivariate joint model, you have the option to
#'   specify a list of prior distributions, however the elements of the list
#'   that correspond to any longitudinal submodel which does not have an 
#'   auxiliary parameter will be ignored. 
#' @param priorEvent_aux The prior distribution for the "auxiliary" parameters
#'   in the event submodel. The "auxiliary" parameters refers to different  
#'   parameters depending on the baseline hazard. For \code{basehaz = "weibull"}
#'   the auxiliary parameter is the Weibull shape parameter. For 
#'   \code{basehaz = "bs"} the auxiliary parameters are the coefficients for the
#'   B-spline approximation to the log baseline hazard.
#'   For \code{basehaz = "piecewise"} the auxiliary parameters are the piecewise
#'   estimates of the log baseline hazard.
#' @param prior_covariance Cannot be \code{NULL}; see \code{\link{priors}} for
#'   more information about the prior distributions on covariance matrices.
#'   Note however that the default prior for covariance matrices in 
#'   \code{stan_jm} is slightly different to that in \code{\link{stan_glmer}} 
#'   (the details of which are described on the \code{\link{priors}} page).
#'   
#' @details The \code{stan_jm} function can be used to fit a joint model (also 
#'   known as a shared parameter model) for longitudinal and time-to-event data 
#'   under a Bayesian framework. The underlying
#'   estimation is carried out using the Bayesian C++ package Stan 
#'   (\url{https://mc-stan.org/}). \cr
#'   \cr 
#'   The joint model may be univariate (with only one longitudinal submodel) or
#'   multivariate (with more than one longitudinal submodel). 
#'   For the longitudinal submodel a (possibly multivariate) generalised linear 
#'   mixed model is assumed with any of the \code{\link[stats]{family}} choices 
#'   allowed by \code{\link[lme4]{glmer}}. If a multivariate joint model is specified 
#'   (by providing a list of formulas in the \code{formulaLong} argument), then
#'   the multivariate longitudinal submodel consists of a multivariate generalized  
#'   linear model (GLM) with group-specific terms that are assumed to be correlated
#'   across the different GLM submodels. That is, within
#'   a grouping factor (for example, patient ID) the group-specific terms are
#'   assumed to be correlated across the different GLM submodels. It is 
#'   possible to specify a different outcome type (for example a different
#'   family and/or link function) for each of the GLM submodels, by providing
#'   a list of \code{\link[stats]{family}} objects in the \code{family} 
#'   argument. Multi-level 
#'   clustered data are allowed, and that additional clustering can occur at a 
#'   level higher than the individual-level (e.g. patients clustered within 
#'   clinics), or at a level lower than the individual-level (e.g. tumor lesions
#'   clustered within patients). If the clustering occurs at a level lower than
#'   the individual, then the user needs to indicate how the lower level 
#'   clusters should be handled when forming the association structure between
#'   the longitudinal and event submodels (see the \code{grp_assoc} argument
#'   described above). \cr
#'   \cr
#'   For the event submodel a parametric
#'   proportional hazards model is assumed. The baseline hazard can be estimated 
#'   using either a cubic B-splines approximation (\code{basehaz = "bs"}, the
#'   default), a Weibull distribution (\code{basehaz = "weibull"}), or a
#'   piecewise constant baseline hazard (\code{basehaz = "piecewise"}).
#'   If the B-spline or piecewise constant baseline hazards are used, 
#'   then the degrees of freedom or the internal knot locations can be 
#'   (optionally) specified. If
#'   the degrees of freedom are specified (through the \code{df} argument) then
#'   the knot locations are automatically generated based on the 
#'   distribution of the observed event times (not including censoring times). 
#'   Otherwise internal knot locations can be specified 
#'   directly through the \code{knots} argument. If neither \code{df} or
#'   \code{knots} is specified, then the default is to set \code{df} equal to 6.
#'   It is not possible to specify both \code{df} and \code{knots}. \cr
#'   \cr
#'   Time-varying covariates are allowed in both the 
#'   longitudinal and event submodels. These should be specified in the data 
#'   in the same way as they normally would when fitting a separate 
#'   longitudinal model using \code{\link[lme4]{lmer}} or a separate 
#'   time-to-event model using \code{\link[survival]{coxph}}. These time-varying
#'   covariates should be exogenous in nature, otherwise they would perhaps 
#'   be better specified as an additional outcome (i.e. by including them as an 
#'   additional longitudinal outcome in the joint model). \cr
#'   \cr
#'   Bayesian estimation of the joint model is performed via MCMC. The Bayesian  
#'   model includes independent priors on the 
#'   regression coefficients for both the longitudinal and event submodels, 
#'   including the association parameter(s) (in much the same way as the
#'   regression parameters in \code{\link{stan_glm}}) and
#'   priors on the terms of a decomposition of the covariance matrices of the
#'   group-specific parameters. 
#'   See \code{\link{priors}} for more information about the priors distributions
#'   that are available. \cr
#'   \cr
#'   Gauss-Kronrod quadrature is used to numerically evaluate the integral  
#'   over the cumulative hazard in the likelihood function for the event submodel.
#'   The accuracy of the numerical approximation can be controlled using the
#'   number of quadrature nodes, specified through the \code{qnodes} 
#'   argument. Using a higher number of quadrature nodes will result in a more 
#'   accurate approximation.
#'   
#'   \subsection{Association structures}{
#'   The association structure for the joint model can be based on any of the 
#'   following parameterisations: 
#'     \itemize{
#'       \item current value of the linear predictor in the 
#'         longitudinal submodel (\code{"etavalue"}) 
#'       \item first derivative (slope) of the linear predictor in the 
#'         longitudinal submodel (\code{"etaslope"}) 
#'       \item the area under the curve of the linear predictor in the 
#'         longitudinal submodel (\code{"etaauc"}) 
#'       \item current expected value of the longitudinal submodel 
#'         (\code{"muvalue"})
#'       \item the area under the curve of the expected value from the 
#'         longitudinal submodel (\code{"muauc"})
#'       \item shared individual-level random effects (\code{"shared_b"}) 
#'       \item shared individual-level random effects which also incorporate 
#'         the corresponding fixed effect as well as any corresponding 
#'         random effects for clustering levels higher than the individual)
#'         (\code{"shared_coef"})
#'       \item interactions between association terms and observed data/covariates
#'         (\code{"etavalue_data"}, \code{"etaslope_data"}, \code{"muvalue_data"}, 
#'         \code{"muslope_data"}). These are described further below.
#'       \item interactions between association terms corresponding to different 
#'         longitudinal outcomes in a multivariate joint model 
#'         (\code{"etavalue_etavalue(#)"}, \code{"etavalue_muvalue(#)"},
#'         \code{"muvalue_etavalue(#)"}, \code{"muvalue_muvalue(#)"}). These
#'         are described further below.      
#'       \item no association structure (equivalent to fitting separate 
#'         longitudinal and event models) (\code{"null"} or \code{NULL}) 
#'     }
#'   More than one association structure can be specified, however,
#'   not all possible combinations are allowed.   
#'   Note that for the lagged association structures baseline values (time = 0) 
#'   are used for the instances 
#'   where the time lag results in a time prior to baseline. When using the 
#'   \code{"etaauc"} or \code{"muauc"} association structures, the area under
#'   the curve is evaluated using Gauss-Kronrod quadrature with 15 quadrature 
#'   nodes. By default, \code{"shared_b"} and \code{"shared_coef"} contribute 
#'   all random effects to the association structure; however, a subset of the 
#'   random effects can be chosen by specifying their indices between parentheses 
#'   as a suffix, for example, \code{"shared_b(1)"} or \code{"shared_b(1:3)"} or 
#'   \code{"shared_b(1,2,4)"}, and so on. \cr
#'   \cr 
#'   In addition, several association terms (\code{"etavalue"}, \code{"etaslope"},
#'   \code{"muvalue"}, \code{"muslope"}) can be interacted with observed 
#'   data/covariates. To do this, use the association term's main handle plus a
#'   suffix of \code{"_data"} then followed by the model matrix formula in 
#'   parentheses. For example if we had a variable in our dataset for gender 
#'   named \code{sex} then we might want to obtain different estimates for the 
#'   association between the current slope of the marker and the risk of the 
#'   event for each gender. To do this we would specify 
#'   \code{assoc = c("etaslope", "etaslope_data(~ sex)")}. \cr
#'   \cr
#'   It is also possible, when fitting  a multivariate joint model, to include 
#'   interaction terms between the association terms themselves (this only
#'   applies for interacting \code{"etavalue"} or \code{"muvalue"}). For example, 
#'   if we had a joint model with two longitudinal markers, we could specify 
#'   \code{assoc = list(c("etavalue", "etavalue_etavalue(2)"), "etavalue")}.
#'   The first element of list says we want to use the value of the linear
#'   predictor for the first marker, as well as it's interaction with the
#'   value of the linear predictor for the second marker. The second element of 
#'   the list says we want to also include the expected value of the second marker 
#'   (i.e. as a "main effect"). Therefore, the linear predictor for the event 
#'   submodel would include the "main effects" for each marker as well as their
#'   interaction. \cr
#'   \cr
#'   There are additional examples in the \strong{Examples} section below.
#'   }
#' 
#' @return A \link[=stanreg-objects]{stanjm} object is returned.
#' 
#' @seealso \code{\link{stanreg-objects}}, \code{\link{stanmvreg-methods}}, 
#'   \code{\link{print.stanmvreg}}, \code{\link{summary.stanmvreg}},
#'   \code{\link{posterior_traj}}, \code{\link{posterior_survfit}}, 
#'   \code{\link{posterior_predict}}, \code{\link{posterior_interval}},
#'   \code{\link{pp_check}}, \code{\link{ps_check}}, \code{\link{stan_mvmer}}.
#' 
#' @examples
#' if (.Platform$OS.type != "windows" || .Platform$r_arch !="i386") {
#' \donttest{
#' 
#' #####
#' # Univariate joint model, with association structure based on the 
#' # current value of the linear predictor
#' f1 <- stan_jm(formulaLong = logBili ~ year + (1 | id), 
#'               dataLong = pbcLong,
#'               formulaEvent = Surv(futimeYears, death) ~ sex + trt, 
#'               dataEvent = pbcSurv,
#'               time_var = "year",
#'               # this next line is only to keep the example small in size!
#'               chains = 1, cores = 1, seed = 12345, iter = 1000)
#' print(f1) 
#' summary(f1) 
#'         
#' #####
#' # Univariate joint model, with association structure based on the 
#' # current value and slope of the linear predictor
#' f2 <- stan_jm(formulaLong = logBili ~ year + (year | id), 
#'               dataLong = pbcLong,
#'               formulaEvent = Surv(futimeYears, death) ~ sex + trt, 
#'               dataEvent = pbcSurv,
#'               assoc = c("etavalue", "etaslope"),
#'               time_var = "year",
#'               chains = 1, cores = 1, seed = 12345, iter = 1000)
#' print(f2)  
#' 
#' #####
#' # Univariate joint model, with association structure based on the 
#' # lagged value of the linear predictor, where the lag is 2 time 
#' # units (i.e. 2 years in this example)
#' f3 <- stan_jm(formulaLong = logBili ~ year + (1 | id), 
#'               dataLong = pbcLong,
#'               formulaEvent = Surv(futimeYears, death) ~ sex + trt, 
#'               dataEvent = pbcSurv,
#'               time_var = "year",
#'               assoc = "etavalue", lag_assoc = 2,
#'               chains = 1, cores = 1, seed = 12345, iter = 1000)
#' print(f3) 
#' 
#' #####
#' # Univariate joint model, where the association structure includes 
#' # interactions with observed data. Here we specify that we want to use 
#' # an association structure based on the current value of the linear 
#' # predictor from the longitudinal submodel (i.e. "etavalue"), but we 
#' # also want to interact this with the treatment covariate (trt) from
#' # pbcLong data frame, so that we can estimate a different association 
#' # parameter (i.e. estimated effect of log serum bilirubin on the log 
#' # hazard of death) for each treatment group
#' f4 <- stan_jm(formulaLong = logBili ~ year + (1 | id), 
#'               dataLong = pbcLong,
#'               formulaEvent = Surv(futimeYears, death) ~ sex + trt, 
#'               dataEvent = pbcSurv,
#'               time_var = "year",
#'               assoc = c("etavalue", "etavalue_data(~ trt)"),
#'               chains = 1, cores = 1, seed = 12345, iter = 1000)
#' print(f4)
#' 
#' ######
#' # Multivariate joint model, with association structure based 
#' # on the current value and slope of the linear predictor in the 
#' # first longitudinal submodel and the area under the marker 
#' # trajectory for the second longitudinal submodel
#' mv1 <- stan_jm(
#'         formulaLong = list(
#'           logBili ~ year + (1 | id), 
#'           albumin ~ sex + year + (year | id)),
#'         dataLong = pbcLong,
#'         formulaEvent = Surv(futimeYears, death) ~ sex + trt, 
#'         dataEvent = pbcSurv,
#'         assoc = list(c("etavalue", "etaslope"), "etaauc"), 
#'         time_var = "year",
#'         chains = 1, cores = 1, seed = 12345, iter = 100)
#' print(mv1)
#' 
#' #####
#' # Multivariate joint model, where the association structure is formed by 
#' # including the expected value of each longitudinal marker (logBili and 
#' # albumin) in the linear predictor of the event submodel, as well as their 
#' # interaction effect (i.e. the interaction between the two "etavalue" terms). 
#' # Note that whether such an association structure based on a marker by 
#' # marker interaction term makes sense will depend on the context of your 
#' # application -- here we just show it for demostration purposes).
#' mv2 <- stan_jm(
#'         formulaLong = list(
#'           logBili ~ year + (1 | id), 
#'           albumin ~ sex + year + (year | id)),
#'         dataLong = pbcLong,
#'         formulaEvent = Surv(futimeYears, death) ~ sex + trt, 
#'         dataEvent = pbcSurv,
#'         assoc = list(c("etavalue", "etavalue_etavalue(2)"), "etavalue"),
#'         time_var = "year", 
#'         chains = 1, cores = 1, seed = 12345, iter = 100)
#'         
#' #####
#' # Multivariate joint model, with one bernoulli marker and one
#' # Gaussian marker. We will artificially create the bernoulli
#' # marker by dichotomising log serum bilirubin
#' pbcLong$ybern <- as.integer(pbcLong$logBili >= mean(pbcLong$logBili))
#' mv3 <- stan_jm(
#'         formulaLong = list(
#'           ybern ~ year + (1 | id), 
#'           albumin ~ sex + year + (year | id)),
#'         dataLong = pbcLong,
#'         formulaEvent = Surv(futimeYears, death) ~ sex + trt, 
#'         dataEvent = pbcSurv,
#'         family = list(binomial, gaussian),
#'         time_var = "year", 
#'         chains = 1, cores = 1, seed = 12345, iter = 1000)
#' }
#' }
#' 
extract_standata <- function(formulaLong, dataLong, formulaEvent, dataEvent, time_var, 
                    id_var, family = gaussian, assoc = "etavalue", 
                    lag_assoc = 0, grp_assoc, scale_assoc = NULL, epsilon = 1E-5,
                    basehaz = c("bs", "weibull", "piecewise"), basehaz_ops,
                    qnodes = 15, init = "prefit", weights,	
                    priorLong = normal(autoscale=TRUE), priorLong_intercept = normal(autoscale=TRUE), 
                    priorLong_aux = cauchy(0, 5, autoscale=TRUE), priorEvent = normal(autoscale=TRUE), 
                    priorEvent_intercept = normal(autoscale=TRUE), priorEvent_aux = cauchy(autoscale=TRUE),
                    priorEvent_assoc = normal(autoscale=TRUE), prior_covariance = lkj(autoscale=TRUE), 
                    prior_PD = FALSE, algorithm = c("sampling", "meanfield", "fullrank"), 
                    adapt_delta = NULL, max_treedepth = 10L, QR = FALSE, 
                    sparse = FALSE, ...) {
  
  #-----------------------------
  # Pre-processing of arguments
  #-----------------------------  
  
  # Set seed if specified
  dots <- list(...)
  if ("seed" %in% names(dots))
    set.seed(dots$seed)
  
  algorithm <- match.arg(algorithm)
  basehaz   <- match.arg(basehaz)
  
  if (missing(basehaz_ops)) basehaz_ops <- NULL
  if (missing(weights))     weights     <- NULL
  if (missing(id_var))      id_var      <- NULL
  if (missing(time_var))    time_var    <- NULL
  if (missing(grp_assoc))   grp_assoc   <- NULL

  if (!is.null(weights)) 
    stop("'weights' are not yet implemented.")
  if (QR)               
    stop("'QR' decomposition is not yet implemented.")
  if (sparse)
    stop("'sparse' option is not yet implemented.")
  
  if (is.null(time_var))
    stop("'time_var' must be specified.")

  # Formula
  formulaLong <- validate_arg(formulaLong, "formula"); M <- length(formulaLong)
	if (M > 3L)
	  stop("'stan_jm' is currently limited to a maximum of 3 longitudinal outcomes.")
  
  # Data
  dataLong <- validate_arg(dataLong, "data.frame", validate_length = M)  
  dataEvent <- as.data.frame(dataEvent)

  # Family
  ok_family_classes <- c("function", "family", "character")
  ok_families <- c("binomial", "gaussian", "Gamma", 
                   "inverse.gaussian", "poisson", "neg_binomial_2")
  family <- validate_arg(family, ok_family_classes, validate_length = M)
  family <- lapply(family, validate_famlink, ok_families)
  
  # Assoc
  ok_assoc_classes <- c("NULL", "character")
  assoc <- validate_arg(assoc, ok_assoc_classes, validate_length = M)

  # Is priorLong* already a list?
  priorLong <- broadcast_prior(priorLong, M)
  priorLong_intercept <- broadcast_prior(priorLong_intercept, M)
  priorLong_aux <- broadcast_prior(priorLong_aux, M)
 
  #-----------
  # Fit model
  #-----------
  
  stanfit <- extract_standata.fit(formulaLong = formulaLong, dataLong = dataLong, 
                         formulaEvent = formulaEvent, dataEvent = dataEvent, 
                         time_var = time_var, id_var = id_var, family = family,
                         assoc = assoc, lag_assoc = lag_assoc, grp_assoc = grp_assoc, 
                         epsilon = epsilon, basehaz = basehaz, basehaz_ops = basehaz_ops, 
                         qnodes = qnodes, init = init, weights = weights, scale_assoc = scale_assoc,
                         priorLong = priorLong, 
                         priorLong_intercept = priorLong_intercept, 
                         priorLong_aux = priorLong_aux, 
                         priorEvent = priorEvent, 
                         priorEvent_intercept = priorEvent_intercept, 
                         priorEvent_aux = priorEvent_aux, 
                         priorEvent_assoc = priorEvent_assoc, 
                         prior_covariance = prior_covariance, prior_PD = prior_PD, 
                         algorithm = algorithm, adapt_delta = adapt_delta, 
                         max_treedepth = max_treedepth, QR = QR, sparse = sparse, ...)
  return(stanfit)
  # if (algorithm != "optimizing" && !is(stanfit, "stanfit")) return(stanfit)
  # y_mod <- attr(stanfit, "y_mod")
  # e_mod <- attr(stanfit, "e_mod")
  # a_mod <- attr(stanfit, "a_mod")
  # cnms  <- attr(stanfit, "cnms")
  # flevels <- attr(stanfit, "flevels")
  # assoc <- attr(stanfit, "assoc")
  # scale_assoc <- attr(stanfit, "scale_assoc")
  # id_var <- attr(stanfit, "id_var")
  # basehaz    <- attr(stanfit, "basehaz")
  # grp_stuff  <- attr(stanfit, "grp_stuff")
  # prior_info <- attr(stanfit, "prior_info")
  # stanfit <- drop_attributes(stanfit, "y_mod", "e_mod", "a_mod", "cnms", 
  #                            "flevels", "assoc", "id_var", "basehaz", 
  #                            "grp_stuff", "prior_info","scale_assoc")
  # 
  # terms <- c(fetch(y_mod, "terms"), list(terms(e_mod$mod)))
  # n_yobs <- fetch_(y_mod, "x", "N")
  # n_grps <- sapply(flevels, n_distinct)
  # n_subjects <- e_mod$Npat
  # 
  # fit <- nlist(stanfit, formula = c(formulaLong, formulaEvent), family,
  #              id_var, time_var, weights, scale_assoc, qnodes, basehaz, assoc,
  #              M, cnms, flevels, n_grps, n_subjects, n_yobs, epsilon,
  #              algorithm, terms, glmod = y_mod, survmod = e_mod, 
  #              assocmod = a_mod, grp_stuff, dataLong, dataEvent,
  #              prior.info = prior_info, stan_function = "stan_jm", 
  #              call = match.call(expand.dots = TRUE))
  # 
  # out <- stanmvreg(fit)
  # return(out)
}

# Part of the rstanarm package for estimating model parameters
# Copyright (C) 2013, 2014, 2015, 2016, 2017 Trustees of Columbia University
# Copyright (C) 2016, 2017 Sam Brilleman
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# Internal model fitting function for models estimated using 
# \code{stan_mvmer} or \code{stan_jm}.
# 
# See \code{stan_jm} for a description of the arguments to the 
# \code{stan_jm.fit} function call.
#
extract_standata.fit <- function(formulaLong = NULL, dataLong = NULL, formulaEvent = NULL, 
                        dataEvent = NULL, time_var, id_var,  family = gaussian, 
                        assoc = "etavalue", lag_assoc = 0, grp_assoc, scale_assoc = NULL,
                        epsilon = 1E-5, basehaz = c("bs", "weibull", "piecewise"),
                        basehaz_ops, qnodes = 15, init = "prefit", weights,
                        priorLong = normal(autoscale=TRUE), priorLong_intercept = normal(autoscale=TRUE), 
                        priorLong_aux = cauchy(0, 5, autoscale=TRUE), priorEvent = normal(autoscale=TRUE), 
                        priorEvent_intercept = normal(autoscale=TRUE), priorEvent_aux = cauchy(autoscale=TRUE),
                        priorEvent_assoc = normal(autoscale=TRUE), prior_covariance = lkj(autoscale=TRUE), prior_PD = FALSE,
                        algorithm = c("sampling", "meanfield", "fullrank"), 
                        adapt_delta = NULL, max_treedepth = 10L, 
                        QR = FALSE, sparse = FALSE, ...) {
  
  #-----------------------------
  # Pre-processing of arguments
  #-----------------------------  
  
  if (!requireNamespace("survival"))
    stop("the 'survival' package must be installed to use this function.")
  
  # Set seed if specified
  dots <- list(...)
  if ("seed" %in% names(dots))
    set.seed(dots$seed)
  
  algorithm <- match.arg(algorithm)
  basehaz   <- match.arg(basehaz)
  
  if (missing(basehaz_ops)) basehaz_ops <- NULL
  if (missing(weights))     weights     <- NULL
  if (missing(id_var))      id_var      <- NULL
  if (missing(time_var))    time_var    <- NULL
  if (missing(grp_assoc))   grp_assoc   <- NULL
  
  if (!is.null(weights)) 
    stop("'weights' are not yet implemented.")
  if (QR)               
    stop("'QR' decomposition is not yet implemented.")
  if (sparse)
    stop("'sparse' option is not yet implemented.")
  
  # Error if args not supplied together
  supplied_together(formulaLong, dataLong, error = TRUE)
  supplied_together(formulaEvent, dataEvent, error = TRUE)
  
  # Determine whether a joint longitudinal-survival model was specified
  is_jm <- supplied_together(formulaLong, formulaEvent)
  stub <- if (is_jm) "Long" else "y"
  
  if (is_jm && is.null(time_var))
    stop("'time_var' must be specified.")
  
  # Formula
  formulaLong <- validate_arg(formulaLong, "formula"); M <- length(formulaLong)
  
  # Data
  dataLong <- validate_arg(dataLong, "data.frame", validate_length = M)  
  if (is_jm)
    dataEvent <- as.data.frame(dataEvent)
  
  # Family
  ok_classes <- c("function", "family", "character")
  ok_families <- c("binomial", "gaussian", "Gamma", 
                   "inverse.gaussian", "poisson", "neg_binomial_2")
  family <- validate_arg(family, ok_classes, validate_length = M)
  family <- lapply(family, validate_famlink, ok_families)
  family <- lapply(family, append_mvmer_famlink)
  
  # Observation weights
  has_weights <- !is.null(weights)
  
  # Priors
  priorLong <- broadcast_prior(priorLong, M)
  priorLong_intercept <- broadcast_prior(priorLong_intercept, M)
  priorLong_aux <- broadcast_prior(priorLong_aux, M)
  
  #--------------------------
  # Longitudinal submodel(s)
  #--------------------------
  
  # Info for separate longitudinal submodels
  y_mod <- xapply(formulaLong, dataLong, family, FUN = handle_y_mod)
  
  # Construct single cnms list for all longitudinal submodels
  y_cnms  <- fetch(y_mod, "z", "group_cnms")
  cnms <- get_common_cnms(y_cnms, stub = stub)
  cnms_nms <- names(cnms)
  if (length(cnms_nms) > 2L)
    stop("A maximum of 2 grouping factors are allowed.")
  
  # Construct single list with unique levels for each grouping factor
  y_flist <- fetch(y_mod, "z", "group_list")
  flevels <- get_common_flevels(y_flist)
  
  # Ensure id_var is a valid grouping factor in all submodels
  if (is_jm) {
    id_var <- check_id_var(id_var, y_cnms, y_flist)
    id_list <- check_id_list(id_var, y_flist)
    if (!is.null(weights))
      weights <- check_weights(weights, id_var)
  }
  
  # Observation weights
  y_weights <- lapply(y_mod, handle_weights, weights, id_var)
  
  #----------- Prior distributions -----------# 
  
  # Valid prior distributions
  ok_dists <- nlist("normal", student_t = "t", "cauchy", "hs", "hs_plus", 
                    "laplace", "lasso")  # disallow product normal
  ok_intercept_dists <- ok_dists[1:3]
  ok_aux_dists <- c(ok_dists[1:3], exponential = "exponential")
  ok_covariance_dists <- c("decov", "lkj")
  
  y_vecs <- fetch(y_mod, "y", "y")     # used in autoscaling
  x_mats <- fetch(y_mod, "x", "xtemp") # used in autoscaling
  
  # Note: *_user_prior_*_stuff objects are stored unchanged for constructing 
  # prior_summary, while *_prior_*_stuff objects are autoscaled
  
  # Priors for longitudinal submodels
  y_links <- fetch(y_mod, "family", "link")
  y_user_prior_stuff <- y_prior_stuff <- 
    xapply(priorLong, nvars = fetch(y_mod, "x", "K"), link = y_links,
           FUN = handle_glm_prior, 
           args = list(default_scale = 2.5, ok_dists = ok_dists))
  
  y_user_prior_intercept_stuff <- y_prior_intercept_stuff <- 
    xapply(priorLong_intercept, link = y_links, 
           FUN = handle_glm_prior,
           args = list(nvars = 1, default_scale = 10, 
                       ok_dists = ok_intercept_dists))
  
  y_user_prior_aux_stuff <- y_prior_aux_stuff <- 
    xapply(priorLong_aux, FUN = handle_glm_prior, 
           args = list(nvars = 1, default_scale = 5, link = NULL, 
                       ok_dists = ok_aux_dists))  
  
  b_user_prior_stuff <- b_prior_stuff <- handle_cov_prior(
    prior_covariance, cnms = cnms, ok_dists = ok_covariance_dists)
  
  # Autoscaling of priors
  y_prior_stuff <- 
    xapply(y_prior_stuff, response = y_vecs, predictors = x_mats, 
           family = family, FUN = autoscale_prior)
  y_prior_intercept_stuff <- 
    xapply(y_prior_intercept_stuff, response = y_vecs,
           family = family, FUN = autoscale_prior)
  y_prior_aux_stuff <- 
    xapply(y_prior_aux_stuff, response = y_vecs,
           family = family, FUN = autoscale_prior)
  if (b_prior_stuff$prior_dist_name == "lkj") { # autoscale priors for ranef sds
    b_prior_stuff <- split_cov_prior(b_prior_stuff, cnms = cnms, submodel_cnms = y_cnms)
    b_prior_stuff <- xapply(
      cnms_nms, FUN = function(nm) {
        z_mats <- fetch(y_mod, "z", "z", nm)
        xapply(b_prior_stuff[[nm]], response = y_vecs, predictors = z_mats, 
               family = family, FUN = autoscale_prior)
      })
  } 
  
  #----------- Data for export to Stan -----------# 
  
  standata <- list(
    M = as.integer(M), 
    has_weights  = as.integer(!all(lapply(weights, is.null))),
    family = fetch_array(y_mod, "family", "mvmer_family"),
    link   = fetch_array(y_mod, "family", "mvmer_link"),
    weights = as.array(numeric(0)), # not yet implemented
    prior_PD = as.integer(prior_PD)
  )  
  
  # Offset
  Y_offset <- fetch(y_mod, "offset", pad_length = 3)
  standata$has_offset <- has_offset <-
    fetch_array(y_mod, "has_offset", pad_length = 3)
  standata$y1_offset <- if (has_offset[1]) Y_offset[[1]] else as.array(integer(0))  
  standata$y2_offset <- if (has_offset[2]) Y_offset[[2]] else as.array(integer(0))  
  standata$y3_offset <- if (has_offset[3]) Y_offset[[3]] else as.array(integer(0)) 
  
  # Dimensions
  standata$has_aux <- 
    fetch_array(y_mod, "has_aux", pad_length = 3)
  standata$resp_type <- 
    fetch_array(y_mod, "y", "resp_type", pad_length = 3)
  standata$intercept_type <- 
    fetch_array(y_mod, "intercept_type", "number", pad_length = 3)
  standata$yNobs <- 
    fetch_array(y_mod, "x", "N", pad_length = 3)
  standata$yNeta <- 
    fetch_array(y_mod, "x", "N", pad_length = 3) # same as Nobs for stan_mvmer
  standata$yK <- 
    fetch_array(y_mod, "x", "K", pad_length = 3)
  
  # Response vectors
  Y_integer <- fetch(y_mod, "y", "integer")
  standata$yInt1 <- if (M > 0) Y_integer[[1]] else as.array(integer(0))  
  standata$yInt2 <- if (M > 1) Y_integer[[2]] else as.array(integer(0))  
  standata$yInt3 <- if (M > 2) Y_integer[[3]] else as.array(integer(0)) 
  
  Y_real <- fetch(y_mod, "y", "real")
  standata$yReal1 <- if (M > 0) Y_real[[1]] else as.array(double(0)) 
  standata$yReal2 <- if (M > 1) Y_real[[2]] else as.array(double(0)) 
  standata$yReal3 <- if (M > 2) Y_real[[3]] else as.array(double(0)) 
  
  # Population level design matrices
  X <- fetch(y_mod, "x", "xtemp")
  standata$yX1 <- if (M > 0) X[[1]] else matrix(0,0,0)
  standata$yX2 <- if (M > 1) X[[2]] else matrix(0,0,0)
  standata$yX3 <- if (M > 2) X[[3]] else matrix(0,0,0)
  
  X_bar <- fetch(y_mod, "x", "x_bar")
  standata$yXbar1 <- if (M > 0) as.array(X_bar[[1]]) else as.array(double(0))
  standata$yXbar2 <- if (M > 1) as.array(X_bar[[2]]) else as.array(double(0))
  standata$yXbar3 <- if (M > 2) as.array(X_bar[[3]]) else as.array(double(0))
  
  # Data for group specific terms - group factor 1
  b1_varname <- cnms_nms[[1L]] # name of group factor 1
  b1_nvars <- fetch_(y_mod, "z", "nvars", b1_varname, 
                     null_to_zero = TRUE, pad_length = 3)
  b1_ngrps <- fetch_(y_mod, "z", "ngrps", b1_varname)
  if (!n_distinct(b1_ngrps) == 1L)
    stop("The number of groups for the grouping factor '", 
         b1_varname, "' should be the same in all submodels.")
  
  standata$bN1 <- b1_ngrps[[1L]] + 1L # add padding for _NEW_ group
  standata$bK1 <- sum(b1_nvars)
  standata$bK1_len <- as.array(b1_nvars)
  standata$bK1_idx <- get_idx_array(b1_nvars)
  
  Z1 <- fetch(y_mod, "z", "z", b1_varname)
  Z1 <- lapply(Z1, transpose)
  Z1 <- lapply(Z1, convert_null, "matrix")
  standata$y1_Z1 <- if (M > 0) Z1[[1L]] else matrix(0,0,0)
  standata$y2_Z1 <- if (M > 1) Z1[[2L]] else matrix(0,0,0)
  standata$y3_Z1 <- if (M > 2) Z1[[3L]] else matrix(0,0,0)
  
  Z1_id <- fetch(y_mod, "z", "group_list", b1_varname)
  Z1_id <- lapply(Z1_id, groups)
  Z1_id <- lapply(Z1_id, convert_null, "arrayinteger")
  standata$y1_Z1_id <- if (M > 0) Z1_id[[1L]] else as.array(integer(0))
  standata$y2_Z1_id <- if (M > 1) Z1_id[[2L]] else as.array(integer(0))
  standata$y3_Z1_id <- if (M > 2) Z1_id[[3L]] else as.array(integer(0))
  
  # Data for group specific terms - group factor 2
  if (length(cnms) > 1L) {
    # model has a second grouping factor
    b2_varname <- cnms_nms[[2L]] # name of group factor 2
    b2_nvars <- fetch_(y_mod, "z", "nvars", b2_varname, 
                       null_to_zero = TRUE, pad_length = 3)
    b2_ngrps <- fetch_(y_mod, "z", "ngrps", b2_varname)
    if (!n_distinct(b2_ngrps) == 1L)
      stop("The number of groups for the grouping factor '", 
           b2_varname, "' should be the same in all submodels.")
    standata$bN2 <- b2_ngrps[[1L]] + 1L # add padding for _NEW_ group
    standata$bK2 <- sum(b2_nvars)
    standata$bK2_len <- as.array(b2_nvars)
    standata$bK2_idx <- get_idx_array(b2_nvars)
    
    Z2 <- fetch(y_mod, "z", "z", b2_varname)
    Z2 <- lapply(Z2, transpose)
    Z2 <- lapply(Z2, convert_null, "matrix")
    standata$y1_Z2 <- if (M > 0) Z2[[1L]] else matrix(0,0,0)
    standata$y2_Z2 <- if (M > 1) Z2[[2L]] else matrix(0,0,0)
    standata$y3_Z2 <- if (M > 2) Z2[[3L]] else matrix(0,0,0)
    
    Z2_id <- fetch(y_mod, "z", "group_list", b2_varname)
    Z2_id <- lapply(Z2_id, groups)
    Z2_id <- lapply(Z2_id, convert_null, "arrayinteger")
    standata$y1_Z2_id <- if (M > 0) Z2_id[[1L]] else as.array(integer(0))
    standata$y2_Z2_id <- if (M > 1) Z2_id[[2L]] else as.array(integer(0))
    standata$y3_Z2_id <- if (M > 2) Z2_id[[3L]] else as.array(integer(0))
    
  } else {
    # no second grouping factor
    standata$bN2 <- 0L
    standata$bK2 <- 0L
    standata$bK2_len <- as.array(rep(0,3L))
    standata$bK2_idx <- get_idx_array(rep(0,3L))
    standata$y1_Z2 <- matrix(0,0,0)
    standata$y2_Z2 <- matrix(0,0,0)
    standata$y3_Z2 <- matrix(0,0,0)
    standata$y1_Z2_id <- as.array(integer(0))
    standata$y2_Z2_id <- as.array(integer(0))
    standata$y3_Z2_id <- as.array(integer(0))
  }
  
  # Priors
  standata$y_prior_dist_for_intercept <- 
    fetch_array(y_prior_intercept_stuff, "prior_dist")  
  standata$y_prior_mean_for_intercept <- 
    fetch_array(y_prior_intercept_stuff, "prior_mean")
  standata$y_prior_scale_for_intercept <- 
    fetch_array(y_prior_intercept_stuff, "prior_scale")
  standata$y_prior_df_for_intercept <- 
    fetch_array(y_prior_intercept_stuff, "prior_df")
  
  standata$y_prior_dist_for_aux <-
    fetch_array(y_prior_aux_stuff, "prior_dist")
  standata$y_prior_mean_for_aux <- 
    fetch_array(y_prior_aux_stuff, "prior_mean")
  standata$y_prior_scale_for_aux <- 
    fetch_array(y_prior_aux_stuff, "prior_scale")
  standata$y_prior_df_for_aux <- 
    fetch_array(y_prior_aux_stuff, "prior_df")
  
  standata$y_prior_dist <- 
    fetch_array(y_prior_stuff, "prior_dist", pad_length = 3)
  
  prior_mean <- fetch(y_prior_stuff, "prior_mean")
  standata$y_prior_mean1 <- if (M > 0) prior_mean[[1]] else as.array(double(0))
  standata$y_prior_mean2 <- if (M > 1) prior_mean[[2]] else as.array(double(0))
  standata$y_prior_mean3 <- if (M > 2) prior_mean[[3]] else as.array(double(0))
  
  prior_scale <- fetch(y_prior_stuff, "prior_scale")
  standata$y_prior_scale1 <- if (M > 0) as.array(prior_scale[[1]]) else as.array(double(0))
  standata$y_prior_scale2 <- if (M > 1) as.array(prior_scale[[2]]) else as.array(double(0))
  standata$y_prior_scale3 <- if (M > 2) as.array(prior_scale[[3]]) else as.array(double(0))
  
  prior_df <- fetch(y_prior_stuff, "prior_df")
  standata$y_prior_df1 <- if (M > 0) prior_df[[1]] else as.array(double(0))
  standata$y_prior_df2 <- if (M > 1) prior_df[[2]] else as.array(double(0))
  standata$y_prior_df3 <- if (M > 2) prior_df[[3]] else as.array(double(0))
  
  # hs priors only
  standata$y_global_prior_scale <- fetch_array(y_prior_stuff, "global_prior_scale") 
  standata$y_global_prior_df <- fetch_array(y_prior_stuff, "global_prior_df")
  standata$y_slab_df <- fetch_array(y_prior_stuff, "slab_df")
  standata$y_slab_scale <- fetch_array(y_prior_stuff, "slab_scale")
  
  # Priors for group specific terms
  standata$t <- length(cnms)
  standata$p <- as.array(sapply(cnms, length))
  standata$l <- as.array(
    sapply(cnms_nms, FUN = function(nm) {
      ngrps <- unique(fetch_(y_mod, "z", "ngrps", nm))
      ngrps + 1L # add padding for _NEW_ group
    }))
  standata$q <- sum(standata$p * standata$l)
  
  if (prior_covariance$dist == "decov") {
    
    # data for decov prior
    standata$prior_dist_for_cov <- b_prior_stuff$prior_dist
    standata$b_prior_shape <- b_prior_stuff$prior_shape
    standata$b_prior_scale <- b_prior_stuff$prior_scale
    standata$b_prior_concentration <- b_prior_stuff$prior_concentration
    standata$b_prior_regularization <- b_prior_stuff$prior_regularization
    standata$len_concentration <- length(standata$b_prior_concentration)
    standata$len_regularization <- length(standata$b_prior_regularization)
    standata$len_theta_L <- sum(choose(standata$p, 2), standata$p)
    
    # pass empty lkj data
    standata$b1_prior_scale <- as.array(rep(0L, standata$bK1))
    standata$b2_prior_scale <- as.array(rep(0L, standata$bK2))
    standata$b1_prior_df <- as.array(rep(0L, standata$bK1))
    standata$b2_prior_df <- as.array(rep(0L, standata$bK2))
    standata$b1_prior_regularization <- 1.0
    standata$b2_prior_regularization <- 1.0   
    
  } else if (prior_covariance$dist == "lkj") {
    
    # data for lkj prior
    b1_prior_stuff <- b_prior_stuff[[b1_varname]]
    b1_prior_dist <- fetch_(b1_prior_stuff, "prior_dist")
    b1_prior_scale <- fetch_array(b1_prior_stuff, "prior_scale")
    b1_prior_df <- fetch_array(b1_prior_stuff, "prior_df")
    b1_prior_regularization <- fetch_(b1_prior_stuff, "prior_regularization")
    if (n_distinct(b1_prior_dist) > 1L)
      stop2("Bug found: covariance prior should be the same for all submodels.")
    if (n_distinct(b1_prior_regularization) > 1L) {
      stop2("Bug found: prior_regularization should be the same for all submodels.")
    }
    standata$prior_dist_for_cov <- unique(b1_prior_dist)
    standata$b1_prior_scale <- b1_prior_scale
    standata$b1_prior_df <- b1_prior_df
    standata$b1_prior_regularization <- if (length(b1_prior_regularization))
      unique(b1_prior_regularization) else 1.0
    
    if (standata$bK2 > 0) {
      # model has a second grouping factor
      b2_prior_stuff <- b_prior_stuff[[b2_varname]]
      b2_prior_scale <- fetch_array(b2_prior_stuff, "prior_scale")
      b2_prior_df    <- fetch_array(b2_prior_stuff, "prior_df")
      b2_prior_regularization <- fetch_(b2_prior_stuff, "prior_regularization")
      standata$b2_prior_scale <- b2_prior_scale
      standata$b2_prior_df    <- b2_prior_df
      standata$b2_prior_regularization <- unique(b2_prior_regularization)
    } else {
      # model does not have a second grouping factor
      standata$b2_prior_scale <- as.array(double(0))
      standata$b2_prior_df <- as.array(double(0))
      standata$b2_prior_regularization <- 1.0
    }
    
    # pass empty decov data
    standata$len_theta_L <- 0L
    standata$b_prior_shape <- as.array(rep(0L, standata$t))
    standata$b_prior_scale <- as.array(rep(0L, standata$t))
    standata$len_concentration <- 0L
    standata$len_regularization <- 0L
    standata$b_prior_concentration <- as.array(rep(0L, standata$len_concentration))
    standata$b_prior_regularization <- as.array(rep(0L, standata$len_regularization))   
  }
  
  # Names for longitudinal submodel parameters
  y_intercept_nms <- uapply(1:M, function(m) {
    if (y_mod[[m]]$intercept_type$number > 0) 
      paste0(stub, m, "|(Intercept)") else NULL
  })
  y_beta_nms <- uapply(1:M, function(m) {
    if (!is.null(colnames(X[[m]]))) 
      paste0(stub, m, "|", colnames(X[[m]])) else NULL
  })
  y_aux_nms <- uapply(1:M, function(m) {
    famname_m <- family[[m]]$family
    if (is.gaussian(famname_m)) paste0(stub, m,"|sigma") else
      if (is.gamma(famname_m)) paste0(stub, m,"|shape") else
        if (is.ig(famname_m)) paste0(stub, m,"|lambda") else
          if (is.nb(famname_m)) paste0(stub, m,"|reciprocal_dispersion") else NULL
  })        
  
  # Names for group specific coefficients ("b pars")
  b_nms <- uapply(seq_along(cnms), FUN = function(i) {
    nm <- cnms_nms[i]
    nms_i <- paste(cnms[[i]], nm)
    flevels[[nm]] <- c(gsub(" ", "_", flevels[[nm]]),
                       paste0("_NEW_", nm))
    if (length(nms_i) == 1) {
      paste0(nms_i, ":", flevels[[nm]])
    } else {
      c(t(sapply(nms_i, paste0, ":", flevels[[nm]])))
    }
  })
  
  # Names for Sigma matrix
  Sigma_nms <- get_Sigma_nms(cnms)
  
  #----------------
  # Event submodel
  #----------------
  
  if (is_jm) { # begin jm block
    
    # Fit separate event submodel
    e_mod <- handle_e_mod(formula = formulaEvent, data = dataEvent, 
                          qnodes = qnodes, id_var = id_var, 
                          y_id_list = id_list)
    
    # Baseline hazard
    ok_basehaz <- nlist("weibull", "bs", "piecewise")
    basehaz <- handle_basehaz(basehaz, basehaz_ops, ok_basehaz = ok_basehaz, 
                              eventtime = e_mod$eventtime, status = e_mod$status)
    
    # Observation weights
    e_weights <- handle_weights(e_mod, weights, id_var)
    
    # Check longitudinal observation times are not later than the event time
    lapply(dataLong, FUN = validate_observation_times,  
           eventtime = e_mod$eventtime, id_var = id_var, time_var = time_var)
    
    #----------- Prior distributions -----------# 
    
    # Valid prior distributions
    ok_e_aux_dists <- ok_dists[1:3]
    
    # Note: *_user_prior_*_stuff objects are stored unchanged for constructing 
    # prior_summary, while *_prior_*_stuff objects are autoscaled
    
    # Priors for event submodel
    e_user_prior_stuff <- e_prior_stuff <- 
      handle_glm_prior(priorEvent, nvars = e_mod$K, default_scale = 2.5, 
                       link = NULL, ok_dists = ok_dists)  
    
    e_user_prior_intercept_stuff <- e_prior_intercept_stuff <- 
      handle_glm_prior(priorEvent_intercept, nvars = 1, default_scale = 20,
                       link = NULL, ok_dists = ok_intercept_dists)  
    
    e_user_prior_aux_stuff <- e_prior_aux_stuff <-
      handle_glm_prior(priorEvent_aux, nvars = basehaz$df,
                       default_scale = if (basehaz$type_name == "weibull") 2 else 20,
                       link = NULL, ok_dists = ok_e_aux_dists)
    
    # Autoscaling of priors
    e_prior_stuff <- 
      autoscale_prior(e_prior_stuff, predictors = e_mod$x$x)
    e_prior_intercept_stuff <- 
      autoscale_prior(e_prior_intercept_stuff)
    e_prior_aux_stuff <- 
      autoscale_prior(e_prior_aux_stuff)
    
    #----------- Data for export to Stan -----------# 
    
    # Data and dimensions
    standata$e_K     <- as.integer(e_mod$K)
    standata$Npat    <- as.integer(e_mod$Npat)
    standata$Nevents <- as.integer(e_mod$Nevents)
    standata$qnodes  <- as.integer(qnodes)
    standata$qwts    <- as.array(e_mod$qwts)
    standata$Npat_times_qnodes <- as.integer(e_mod$Npat * qnodes)
    standata$e_times <- as.array(e_mod$cpts)
    standata$nrow_e_Xq <- length(standata$e_times)
    standata$e_has_intercept <- as.integer(basehaz$type_name == "weibull")
    standata$e_Xq    <- e_mod$Xq
    standata$e_xbar  <- as.array(e_mod$Xbar)
    standata$e_weights <- as.array(e_weights)
    standata$e_weights_rep <- as.array(rep(e_weights, times = qnodes))
    
    # Baseline hazard
    standata$basehaz_type <- as.integer(basehaz$type)
    standata$basehaz_df   <- as.integer(basehaz$df)
    standata$basehaz_X <- make_basehaz_X(e_mod$cpts, basehaz)
    standata$norm_const <- e_mod$norm_const    
    
    # Priors
    standata$e_prior_dist              <- e_prior_stuff$prior_dist
    standata$e_prior_dist_for_intercept<- e_prior_intercept_stuff$prior_dist
    standata$e_prior_dist_for_aux      <- e_prior_aux_stuff$prior_dist
    
    # hyperparameters for event submodel priors
    standata$e_prior_mean               <- e_prior_stuff$prior_mean
    standata$e_prior_scale              <- e_prior_stuff$prior_scale
    standata$e_prior_df                 <- e_prior_stuff$prior_df
    standata$e_prior_mean_for_intercept <- c(e_prior_intercept_stuff$prior_mean)
    standata$e_prior_scale_for_intercept<- c(e_prior_intercept_stuff$prior_scale)
    standata$e_prior_df_for_intercept   <- c(e_prior_intercept_stuff$prior_df)
    standata$e_prior_mean_for_aux       <- if (basehaz$type == 1L) as.array(0) else 
      as.array(e_prior_aux_stuff$prior_mean)
    standata$e_prior_scale_for_aux      <- e_prior_aux_stuff$prior_scale
    standata$e_prior_df_for_aux         <- e_prior_aux_stuff$prior_df
    standata$e_global_prior_scale       <- e_prior_stuff$global_prior_scale
    standata$e_global_prior_df          <- e_prior_stuff$global_prior_df
    standata$e_slab_df                  <- e_prior_stuff$slab_df
    standata$e_slab_scale               <- e_prior_stuff$slab_scale
    
    #-----------------------
    # Association structure
    #-----------------------
    
    # Handle association structure
    # !! If order is changed here, then must also change standata$has_assoc !!
    ok_assoc <- c("null", "etavalue","etaslope", "etaauc", "muvalue", 
                  "muslope", "muauc", "shared_b", "shared_coef")
    ok_assoc_data <- ok_assoc[c(2:3,5:6)]
    ok_assoc_interactions <- ok_assoc[c(2,5)]
    
    lag_assoc <- validate_lag_assoc(lag_assoc, M)
    
    assoc <- mapply(assoc, y_mod = y_mod, lag = lag_assoc, FUN = validate_assoc, 
                    MoreArgs = list(ok_assoc = ok_assoc, ok_assoc_data = ok_assoc_data,
                                    ok_assoc_interactions = ok_assoc_interactions, 
                                    id_var = id_var, M = M))
    assoc <- check_order_of_assoc_interactions(assoc, ok_assoc_interactions)
    colnames(assoc) <- paste0("Long", 1:M)
    
    # For each submodel, identify any grouping factors that are
    # clustered within id_var (i.e. lower level clustering)
    ok_grp_assocs <- c("sum", "mean", "min", "max")
    grp_basic <- xapply(FUN = get_basic_grp_info, 
                        cnms  = y_cnms, flist = y_flist,
                        args = list(id_var = id_var))
    grp_stuff <- xapply(FUN = get_extra_grp_info,
                        basic_info = grp_basic, flist = y_flist,
                        args = list(id_var = id_var, grp_assoc = grp_assoc, 
                                    ok_grp_assocs = ok_grp_assocs))
    has_grp <- fetch_(grp_stuff, "has_grp")
    if (any(has_grp)) {
      grp_structure <- fetch(grp_stuff, "grp_list")[has_grp]
      if (n_distinct(grp_structure) > 1L)
        stop2("Any longitudinal submodels with a grouping factor clustered within ",
              "patients must use the same clustering structure; that is, the same ",
              "clustering variable and the same number of units clustered within a ",
              "given patient.")
      ok_assocs_with_grp <- c("etavalue", "etavalue_data", "etaslope", "etaslope_data", 
                              "muvalue", "muvalue_data")
      validate_assoc_with_grp(has_grp = has_grp, assoc = assoc, 
                              ok_assocs_with_grp = ok_assocs_with_grp)
    } else if (!is.null(grp_assoc)) {
      stop2("'grp_assoc' can only be specified when there is a grouping factor ",
            "clustered within patients.")  
    }    
    
    # Return design matrices for evaluating longitudinal submodel quantities
    # at the quadrature points
    auc_qnodes <- 15L
    assoc_as_list <- apply(assoc, 2L, c)
    a_mod <- xapply(data = dataLong, assoc = assoc_as_list, y_mod = y_mod,
                    grp_stuff = grp_stuff, FUN = handle_assocmod, 
                    args = list(ids = e_mod$cids, times = e_mod$cpts, 
                                id_var = id_var, time_var = time_var, 
                                epsilon = epsilon, auc_qnodes = auc_qnodes))
    
    # Number of association parameters
    a_K <- get_num_assoc_pars(assoc, a_mod)
    
    # Association scaling parameter
    a_scale <- validate_scale_assoc(scale_assoc, assoc_as_list)
    
    # Use a stan_mvmer variational bayes model fit for:
    # - obtaining initial values for joint model parameters
    # - obtaining appropriate scaling for priors on association parameters
    vbdots <- list(...)
    dropargs <- c("chains", "cores", "iter", "refresh", "thin", "test_grad", "control")
    for (i in dropargs) 
      vbdots[[i]] <- NULL
    vbpars <- pars_to_monitor(standata, is_jm = FALSE)
    vbargs <- c(list(stanmodels$mvmer, pars = vbpars, data = standata, 
                     algorithm = "meanfield"), vbdots)
    utils::capture.output(init_fit <- suppressWarnings(do.call(rstan::vb, vbargs)))
    init_new_nms <- c(y_intercept_nms, y_beta_nms,
                      if (length(standata$q)) c(paste0("b[", b_nms, "]")),
                      y_aux_nms, paste0("Sigma[", Sigma_nms, "]"),
                      paste0(stub, 1:M, "|mean_PPD"), "log-posterior")
    init_fit@sim$fnames_oi <- init_new_nms
    init_mat <- t(colMeans(as.matrix(init_fit))) # posterior means
    init_nms <- collect_nms(colnames(init_mat), M, stub = "Long")
    init_beta <- lapply(1:M, function(m) init_mat[, init_nms$y[[m]]])
    init_b <- lapply(1:M, function(m) {
      # can drop _NEW_ groups since they are not required for generating
      # the assoc_terms that are used in scaling the priors for 
      # the association parameters (ie. the Zt matrix returned by the 
      # function 'make_assoc_parts_for_stan' will not be padded).
      b <- init_mat[, init_nms$y_b[[m]]]
      b[!grepl("_NEW_", names(b), fixed = TRUE)]
    })
    
    if (is.character(init) && (init =="prefit")) {
      init_means2 <- rstan::get_posterior_mean(init_fit)
      init_nms2 <- rownames(init_means2)
      inits <- generate_init_function(e_mod, standata)()
      
      sel_b1 <- grep(paste0("^z_bMat1\\."), init_nms2)
      if (length(sel_b1))
        inits[["z_bMat1"]] <- matrix(init_means2[sel_b1,], nrow = standata$bK1)
      
      sel_b2 <- grep(paste0("^z_bMat2\\."), init_nms2)
      if (length(sel_b2))
        inits[["z_bMat2"]] <- matrix(init_means2[sel_b2,], nrow = standata$bK2)
      
      sel_bC1 <- grep(paste0("^bCholesky1\\."), init_nms2)
      if (length(sel_bC1) > 1) {
        inits[["bCholesky1"]] <- matrix(init_means2[sel_bC1,], nrow = standata$bK1)
      } else if (length(sel_bC1) == 1) {
        inits[["bCholesky1"]] <- as.array(init_means2[sel_bC1,])
      }
      
      sel_bC2 <- grep(paste0("^bCholesky2\\."), init_nms2)
      if (length(sel_bC2) > 1) {
        inits[["bCholesky2"]] <- matrix(init_means2[sel_bC2,], nrow = standata$bK2)
      } else if (length(sel_bC1) == 1) {
        inits[["bCholesky2"]] <- as.array(init_means2[sel_bC2,])
      }      
      
      sel <- c("yGamma1", "yGamma2", "yGamma3", 
               "z_yBeta1", "z_yBeta2", "z_yBeta3",
               "yAux1_unscaled", "yAux2_unscaled", "yAux3_unscaled", 
               "bSd1", "bSd2", "z_b", "z_T", "rho", "zeta", "tau", 
               "yGlobal1", "yGlobal2", "yGlobal3", 
               "yLocal1", "yLocal2", "yLocal3", 
               "yMix1", "yMix2", "yMix3", 
               "yOol1", "yOol2", "yOol3")
      for (i in sel) {
        sel_i <- grep(paste0("^", i, "\\."), init_nms2)
        if (length(sel_i))
          inits[[i]] <- as.array(init_means2[sel_i,])
      }
      init <- function() inits
    }
    
    #----------- Prior distributions -----------# 
    
    # Priors for association parameters
    e_user_prior_assoc_stuff <- e_prior_assoc_stuff <- 
      handle_glm_prior(priorEvent_assoc, nvars = a_K, default_scale = 2.5,
                       link = NULL, ok_dists = ok_dists)  
    
    # Autoscaling of priors
    if (a_K) {
      e_prior_assoc_stuff <- autoscale_prior(e_prior_assoc_stuff, family = family, 
                                             assoc = assoc, parts = a_mod,
                                             beta = init_beta, b = init_b, 
                                             scale_assoc = a_scale)
    }   
    
    #----------- Data for export to Stan -----------# 
    
    # Dimensions   
    standata$assoc <- as.integer(a_K > 0L) # any association structure, 1 = yes
    standata$a_K   <- as.integer(a_K)      # num association parameters
    
    # Indicator for which components are required to build the association terms
    assoc_uses <- sapply(
      c("etavalue", "etaslope", "etaauc", "muvalue", "muslope", "muauc"), 
      function(x, assoc) {
        nm_check <- switch(x,
                           etavalue = "^eta|^mu",
                           etaslope = "etaslope|muslope",
                           etaauc   = "etaauc|muauc",
                           muvalue  = "muvalue|muslope",
                           muslope  = "muslope",
                           muauc    = "muauc")
        sel <- grep(nm_check, rownames(assoc))
        tmp <- assoc[sel, , drop = FALSE]
        tmp <- pad_matrix(tmp, cols = 3L, value = FALSE)
        as.integer(as.logical(colSums(tmp > 0)))
      }, assoc = assoc)
    standata$assoc_uses <- t(assoc_uses)
    
    # Indexing for desired association types
    # !! Must be careful with corresponding use of indexing in Stan code !!
    # 1 = ev; 2 = es; 3 = ea; 4 = mv; 5 = ms; 6 = ma;
    # 7 = shared_b; 8 = shared_coef;
    # 9 = ev_data; 10 = es_data; 11 = mv_data; 12 = ms_data;
    # 13 = evev; 14 = evmv; 15 = mvev; 16 = mvmv;
    sel <- grep("which|null", rownames(assoc), invert = TRUE)
    standata$has_assoc <- matrix(as.integer(assoc[sel,]), ncol = M) 
    
    # Data for association structure when there is
    # clustering below the patient-level
    standata$has_grp <- as.array(as.integer(has_grp))
    if (any(has_grp)) { # has lower level clustering
      sel <- which(has_grp)[[1L]]
      standata$grp_idx <- attr(a_mod[[sel]], "grp_idx")
      standata$grp_assoc <- switch(grp_assoc, 
                                   sum = 1L,
                                   mean = 2L,
                                   min = 3L,
                                   max = 4L,
                                   0L)
    } else { # no lower level clustering
      standata$grp_idx <- matrix(0L, standata$nrow_e_Xq, 2L)
      standata$grp_assoc <- 0L
    }
    
    # Data for calculating eta, slope, auc in GK quadrature 
    N_tmp <- sapply(a_mod, function(x) NROW(x$mod_eta$xtemp))
    N_tmp <- c(N_tmp, rep(0, 3 - length(N_tmp)))
    standata$nrow_y_Xq <- as.array(as.integer(N_tmp))
    for (m in 1:3) {
      for (i in c("eta", "eps", "auc")) {
        nm_check <- switch(i,
                           eta = "^eta|^mu",
                           eps = "slope",
                           auc = "auc")
        sel <- grep(nm_check, rownames(assoc))
        if (m <= M && any(unlist(assoc[sel,m]))) {
          tmp_stuff <- a_mod[[m]][[paste0("mod_", i)]]
          # fe design matrix at quadpoints
          X_tmp <- tmp_stuff$xtemp
          # re design matrix at quadpoints, group factor 1
          Z1_tmp <- tmp_stuff$z[[cnms_nms[1L]]]
          Z1_tmp <- transpose(Z1_tmp)
          Z1_tmp <- convert_null(Z1_tmp, "matrix")
          Z1_tmp_id <- tmp_stuff$group_list[[cnms_nms[1L]]]
          Z1_tmp_id <- groups(Z1_tmp_id)
          Z1_tmp_id <- convert_null(Z1_tmp_id, "arrayinteger")
          # re design matrix at quadpoints, group factor 1
          if (length(cnms_nms) > 1L) {
            Z2_tmp <- tmp_stuff$z[[cnms_nms[2L]]]
            Z2_tmp <- transpose(Z2_tmp)
            Z2_tmp <- convert_null(Z2_tmp, "matrix")
            Z2_tmp_id <- tmp_stuff$group_list[[cnms_nms[2L]]]
            Z2_tmp_id <- groups(Z2_tmp_id)
            Z2_tmp_id <- convert_null(Z2_tmp_id, "arrayinteger")
          } else {
            Z2_tmp <- matrix(0,standata$bK2_len[m],0) 
            Z2_tmp_id <- as.array(integer(0))
          }
          y_offset_tmp <- if (has_offset[m]) tmp_stuff$offset else as.array(integer(0))
        } else {
          X_tmp  <- matrix(0,0,standata$yK[m])
          Z1_tmp <- matrix(0,standata$bK1_len[m],0) 
          Z2_tmp <- matrix(0,standata$bK2_len[m],0) 
          Z1_tmp_id <- as.array(integer(0)) 
          Z2_tmp_id <- as.array(integer(0)) 
          y_offset_tmp <- as.array(integer(0))
        }
        standata[[paste0("y", m, "_xq_", i)]] <- X_tmp
        standata[[paste0("y", m, "_z1q_", i)]] <- Z1_tmp
        standata[[paste0("y", m, "_z2q_", i)]] <- Z2_tmp
        standata[[paste0("y", m, "_z1q_id_", i)]] <- Z1_tmp_id
        standata[[paste0("y", m, "_z2q_id_", i)]] <- Z2_tmp_id
        standata[[paste0("y", m, "_offset_", i)]] <- y_offset_tmp
      }
    }
    
    # Data for auc association structure
    standata$auc_qnodes <- as.integer(auc_qnodes)
    standata$Npat_times_auc_qnodes <- as.integer(e_mod$Npat * auc_qnodes) 
    nrow_y_Xq_auc <- unique(uapply(a_mod, function(x) {
      nr <- NROW(x$mod_auc$x)
      if (nr > 0) nr else NULL
    }))
    if (length(nrow_y_Xq_auc) > 1L)
      stop2("Bug found: nrows for auc should be the same for all submodels.")
    standata$nrow_y_Xq_auc <- if (!is.null(nrow_y_Xq_auc)) nrow_y_Xq_auc else 0L
    auc_qwts <- uapply(e_mod$cpts, function(x)
      lapply(get_quadpoints(auc_qnodes)$weights, unstandardise_qwts, 0, x))
    standata$auc_qwts <- 
      if (any(standata$assoc_uses[3,] > 0)) as.array(auc_qwts) else double(0)    
    
    # Interactions between association terms and data, with the following objects:
    #   a_K_data: number of columns in y_Xq_data corresponding to each interaction 
    #     type (ie, etavalue, etaslope, muvalue, muslope) for each submodel
    #   idx_q: indexing for the rows of Xq_data that correspond to each submodel, 
    #     since it is formed as a block diagonal matrix
    Xq_data <- fetch(a_mod, "X_bind_data") # design mat for the interactions
    standata$y_Xq_data <- as.array(as.matrix(Matrix::bdiag(Xq_data)))
    standata$a_K_data <- fetch_array(a_mod, "K_data")
    standata$idx_q <- get_idx_array(standata$nrow_y_Xq)
    
    # Interactions between association terms
    standata$which_interactions      <- as.array(unlist(assoc["which_interactions",]))
    standata$size_which_interactions <- c(sapply(assoc["which_interactions",], sapply, length))
    
    # Shared random effects
    standata$which_b_zindex    <- as.array(unlist(assoc["which_b_zindex",]))
    standata$which_coef_zindex <- as.array(unlist(assoc["which_coef_zindex",]))
    standata$which_coef_xindex <- as.array(unlist(assoc["which_coef_xindex",]))
    standata$size_which_b      <- as.array(sapply(assoc["which_b_zindex",    ], length))
    standata$size_which_coef   <- as.array(sapply(assoc["which_coef_zindex", ], length))
    
    # Sum dimensions
    for (i in c("a_K_data", paste0("size_which_", c("b", "coef", "interactions")))) {
      standata[[paste0("sum_", i)]] <- as.integer(sum(standata[[i]]))
    }
    
    # Hyperparameters for assoc parameter priors
    standata$a_prior_dist  <- e_prior_assoc_stuff$prior_dist 
    standata$a_prior_mean  <- e_prior_assoc_stuff$prior_mean
    standata$a_prior_scale <- as.array(e_prior_assoc_stuff$prior_scale)
    standata$a_prior_df    <- e_prior_assoc_stuff$prior_df
    standata$a_global_prior_scale <- e_prior_assoc_stuff$global_prior_scale
    standata$a_global_prior_df    <- e_prior_assoc_stuff$global_prior_df
    standata$a_slab_df            <- e_prior_assoc_stuff$slab_df
    standata$a_slab_scale         <- e_prior_assoc_stuff$slab_scale
    
    # Centering for association terms
    standata$a_xbar <- if (a_K) e_prior_assoc_stuff$a_xbar else numeric(0)    
    
    # Scaling for association terms
    standata$a_scale <- if (a_K) as.array(a_scale) else numeric(0)
    
  } # end jm block
  
  #---------------
  # Prior summary
  #---------------
  
  prior_info <- summarize_jm_prior(
    user_priorLong = y_user_prior_stuff,
    user_priorLong_intercept = y_user_prior_intercept_stuff,
    user_priorLong_aux = y_user_prior_aux_stuff,
    if (is_jm) user_priorEvent = e_user_prior_stuff,
    if (is_jm) user_priorEvent_intercept = e_user_prior_intercept_stuff,
    if (is_jm) user_priorEvent_aux = e_user_prior_aux_stuff,
    if (is_jm) user_priorEvent_assoc = e_user_prior_assoc_stuff,
    user_prior_covariance = prior_covariance,
    b_user_prior_stuff = b_user_prior_stuff,
    b_prior_stuff = b_prior_stuff,
    y_has_intercept = fetch_(y_mod, "x", "has_intercept"),
    y_has_predictors = fetch_(y_mod, "x", "K") > 0,
    if (is_jm) e_has_intercept = standata$e_has_intercept,
    if (is_jm) e_has_predictors = standata$e_K > 0,
    if (is_jm) has_assoc = a_K > 0,
    adjusted_priorLong_scale = fetch(y_prior_stuff, "prior_scale"),
    adjusted_priorLong_intercept_scale = fetch(y_prior_intercept_stuff, "prior_scale"),
    adjusted_priorLong_aux_scale = fetch(y_prior_aux_stuff, "prior_scale"),
    if (is_jm) adjusted_priorEvent_scale = e_prior_stuff$prior_scale,
    if (is_jm) adjusted_priorEvent_intercept_scale = e_prior_intercept_stuff$prior_scale,
    if (is_jm) adjusted_priorEvent_aux_scale = e_prior_aux_stuff$prior_scale,
    if (is_jm) adjusted_priorEvent_assoc_scale = e_prior_assoc_stuff$prior_scale,
    family = family, 
    if (is_jm) basehaz = basehaz,
    stub_for_names = if (is_jm) "Long" else "y"
  )  
  return(standata)
  # #-----------
  # # Fit model
  # #-----------
  # 
  # # call stan() to draw from posterior distribution
  # stanfit <- if (is_jm) stanmodels$jm else stanmodels$mvmer
  # pars <- pars_to_monitor(standata, is_jm = is_jm)
  # if (M == 1L) 
  #   cat("Fitting a univariate", if (is_jm) "joint" else "glmer", "model.\n\n")
  # if (M  > 1L) 
  #   cat("Fitting a multivariate", if (is_jm) "joint" else "glmer", "model.\n\n")
  # 
  # if (algorithm == "sampling") {
  #   cat("Please note the warmup may be much slower than later iterations!\n")             
  #   sampling_args <- set_jm_sampling_args(
  #     object = stanfit,
  #     cnms = cnms,
  #     user_dots = list(...), 
  #     user_adapt_delta = adapt_delta,
  #     user_max_treedepth = max_treedepth,
  #     data = standata, 
  #     pars = pars, 
  #     init = init,
  #     show_messages = FALSE)
  #   stanfit <- do.call(sampling, sampling_args)
  # } else {
  #   # meanfield or fullrank vb
  #   stanfit <- rstan::vb(stanfit, pars = pars, data = standata,
  #                        algorithm = algorithm, ...)    
  # }
  # check <- check_stanfit(stanfit)
  # if (!isTRUE(check)) return(standata)
  # 
  # # Sigma values in stanmat
  # if (prior_covariance$dist == "decov" && standata$len_theta_L)
  #   stanfit <- evaluate_Sigma(stanfit, cnms)
  # 
  # if (is_jm) { # begin jm block
  #   
  #   e_intercept_nms <- "Event|(Intercept)"
  #   e_beta_nms <- if (e_mod$K) paste0("Event|", colnames(e_mod$Xq)) else NULL  
  #   e_aux_nms <- 
  #     if (basehaz$type_name == "weibull") "Event|weibull-shape" else 
  #       if (basehaz$type_name == "bs") paste0("Event|b-splines-coef", seq(basehaz$df)) else
  #         if (basehaz$type_name == "piecewise") paste0("Event|piecewise-coef", seq(basehaz$df)) 
  #   e_assoc_nms <- character()  
  #   for (m in 1:M) {
  #     if (assoc["etavalue",         ][[m]]) e_assoc_nms <- c(e_assoc_nms, paste0("Assoc|Long", m,"|etavalue"))
  #     if (assoc["etavalue_data",    ][[m]]) e_assoc_nms <- c(e_assoc_nms, paste0("Assoc|Long", m,"|etavalue:", colnames(a_mod[[m]][["X_data"]][["etavalue_data"]])))
  #     if (assoc["etavalue_etavalue",][[m]]) e_assoc_nms <- c(e_assoc_nms, paste0("Assoc|Long", m,"|etavalue:Long", assoc["which_interactions",][[m]][["etavalue_etavalue"]], "|etavalue"))
  #     if (assoc["etavalue_muvalue", ][[m]]) e_assoc_nms <- c(e_assoc_nms, paste0("Assoc|Long", m,"|etavalue:Long", assoc["which_interactions",][[m]][["etavalue_muvalue"]],  "|muvalue"))
  #     if (assoc["etaslope",         ][[m]]) e_assoc_nms <- c(e_assoc_nms, paste0("Assoc|Long", m,"|etaslope"))
  #     if (assoc["etaslope_data",    ][[m]]) e_assoc_nms <- c(e_assoc_nms, paste0("Assoc|Long", m,"|etaslope:", colnames(a_mod[[m]][["X_data"]][["etaslope_data"]])))    
  #     if (assoc["etaauc",           ][[m]]) e_assoc_nms <- c(e_assoc_nms, paste0("Assoc|Long", m,"|etaauc"))
  #     if (assoc["muvalue",          ][[m]]) e_assoc_nms <- c(e_assoc_nms, paste0("Assoc|Long", m,"|muvalue"))
  #     if (assoc["muvalue_data",     ][[m]]) e_assoc_nms <- c(e_assoc_nms, paste0("Assoc|Long", m,"|muvalue:", colnames(a_mod[[m]][["X_data"]][["muvalue_data"]])))    
  #     if (assoc["muvalue_etavalue", ][[m]]) e_assoc_nms <- c(e_assoc_nms, paste0("Assoc|Long", m,"|muvalue:Long", assoc["which_interactions",][[m]][["muvalue_etavalue"]], "|etavalue"))
  #     if (assoc["muvalue_muvalue",  ][[m]]) e_assoc_nms <- c(e_assoc_nms, paste0("Assoc|Long", m,"|muvalue:Long", assoc["which_interactions",][[m]][["muvalue_muvalue"]],  "|muvalue"))
  #     if (assoc["muslope",          ][[m]]) e_assoc_nms <- c(e_assoc_nms, paste0("Assoc|Long", m,"|muslope"))
  #     if (assoc["muslope_data",     ][[m]]) e_assoc_nms <- c(e_assoc_nms, paste0("Assoc|Long", m,"|muslope:", colnames(a_mod[[m]][["X_data"]][["muslope_data"]])))    
  #     if (assoc["muauc",            ][[m]]) e_assoc_nms <- c(e_assoc_nms, paste0("Assoc|Long", m,"|muauc"))
  #   }
  #   if (sum(standata$size_which_b)) {
  #     temp_g_nms <- lapply(1:M, FUN = function(m) {
  #       all_nms <- paste0(paste0("Long", m, "|b["), y_mod[[m]]$z$group_cnms[[id_var]], "]")
  #       all_nms[assoc["which_b_zindex",][[m]]]})
  #     e_assoc_nms <- c(e_assoc_nms, paste0("Assoc|", unlist(temp_g_nms)))
  #   }
  #   if (sum(standata$size_which_coef)) {
  #     temp_g_nms <- lapply(1:M, FUN = function(m) {
  #       all_nms <- paste0(paste0("Long", m, "|coef["), y_mod[[m]]$z$group_cnms[[id_var]], "]")
  #       all_nms[assoc["which_coef_zindex",][[m]]]})
  #     e_assoc_nms <- c(e_assoc_nms, paste0("Assoc|", unlist(temp_g_nms)))
  #   }
  #   
  # } # end jm block
  # 
  # new_names <- c(y_intercept_nms,
  #                y_beta_nms,
  #                if (is_jm) e_intercept_nms,
  #                if (is_jm) e_beta_nms,
  #                if (is_jm) e_assoc_nms,                   
  #                if (length(standata$q)) c(paste0("b[", b_nms, "]")),
  #                y_aux_nms,
  #                if (is_jm) e_aux_nms,
  #                paste0("Sigma[", Sigma_nms, "]"),
  #                paste0(stub, 1:M, "|mean_PPD"), 
  #                "log-posterior")
  # stanfit@sim$fnames_oi <- new_names
  # 
  # stanfit_str <- nlist(.Data = stanfit, prior_info, y_mod, cnms, flevels)
  # if (is_jm)
  #   stanfit_str <- c(stanfit_str, nlist(e_mod, a_mod, assoc, basehaz, 
  #                                       id_var, grp_stuff, scale_assoc))
  # 
  # do.call("structure", stanfit_str)
}



