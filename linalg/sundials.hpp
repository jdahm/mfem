// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_SUNDIALS
#define MFEM_SUNDIALS

#include "../config/config.hpp"

#ifdef MFEM_USE_SUNDIALS

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

#include "ode.hpp"
#include "solvers.hpp"

#include <cvode/cvode.h>
#include <arkode/arkode.h>
#include <kinsol/kinsol.h>

#include <cvode/cvode_impl.h>
#include <cvode/cvode_spgmr.h>

// This just hides a warning (to be removed after it's fixed in SUNDIALS).
#ifdef MSG_TIME_INT
#undef MSG_TIME_INT
#endif

#include <arkode/arkode_impl.h>
#include <arkode/arkode_spgmr.h>
#include <kinsol/kinsol_impl.h>
#include <kinsol/kinsol_spgmr.h>

struct KINMemRec;

namespace mfem
{

#ifdef MFEM_USE_NVECTOR_OCCA
/// OCCA NVector content
struct NVOCCAContent {
   OccaVector *vec;
   bool ownVector;
#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif
};
#endif

namespace nv
{

/* N_Vector methods to avoid further specialization */
// TODO Remove dependence on these here

/// Disassociate the existing data and set to v
void SetVector(const N_Vector &nv, Vector &v);

#ifdef MFEM_USE_OCCA
void SetVector(const N_Vector &nv, OccaVector &v);
#endif

///  Return the vector local length
long int GetLength(const N_Vector &nv);

} // namespace nv


/** @brief Abstract base class, wrapping the custom linear solvers interface in
    SUNDIALS' CVODE and ARKODE solvers. */
/** For a given ODE system

        dx/dt = f(x,t)

    the purpose of this class is to facilitate the (approximate) solution of
    linear systems of the form

        (I - γJ) y = b,   J = J(x,t) = df/dx

    for given b, x, t and γ, where γ = GetTimeStep() is a scaled time step. */
class SundialsODELinearSolver
{
public:
   enum {CVODE, ARKODE} type; ///< Is CVODE or ARKODE using this object?

protected:
   SundialsODELinearSolver() { }
   virtual ~SundialsODELinearSolver() { }

   /// Get the current scaled time step, gamma, from @a sundials_mem.
   double GetTimeStep(void *sundials_mem);
   /// Get the TimeDependentOperator associated with @a sundials_mem.
   TimeDependentOperator *GetTimeDependentOperator(void *sundials_mem);

public:
   /** @name Linear solver interface methods.
       These four functions and their parameters are documented in Section 7 of
       http://computation.llnl.gov/sites/default/files/public/cv_guide.pdf
       and Section 7.4 of
       http://computation.llnl.gov/sites/default/files/public/ark_guide.pdf

       The first argument, @a sundials_mem, is one of the pointer types,
       CVodeMem or ARKodeMem, depending on the value of the data member @a type.
   */
   ///@{
   virtual int InitSystem(void *sundials_mem) = 0;
   virtual int SetupSystem(void *sundials_mem, int conv_fail,
                           const Vector &y_pred, const Vector &f_pred,
                           int &jac_cur, Vector &v_temp1,
                           Vector &v_temp2, Vector &v_temp3) = 0;
   virtual int SolveSystem(void *sundials_mem, Vector &b, const Vector &weight,
                           const Vector &y_cur, const Vector &f_cur) = 0;
   virtual int FreeSystem(void *sundials_mem) = 0;
   ///@}
};


/// A base class for the MFEM classes wrapping SUNDIALS' solvers.
/** This class defines some common data and functions used by the SUNDIALS
    solvers, e.g the common @a #sundials_mem pointer and return @a #flag. */
class SundialsSolver
{
protected:
   void *sundials_mem; ///< Pointer to the SUNDIALS mem object.
   mutable int flag;   ///< Flag returned by the last call to SUNDIALS.

   N_Vector y;  ///< Auxiliary N_Vector.

   /// Temporary: Specifies the vector type passed to overloaded TimeDependentOperator::Step method
   static int vectorType;

#ifdef MFEM_USE_MPI
   MPI_Comm sundials_comm;
   bool Parallel() const { return (sundials_comm != MPI_COMM_NULL); }
#else
   bool Parallel() const { return false; }
#endif

   static const double default_rel_tol;
   static const double default_abs_tol;

   // Computes the action of a time-dependent operator.
   /// Callback function used in CVODESolver and ARKODESolver.
   static int ODEMult(realtype t, const N_Vector y,
                      N_Vector ydot, void *td_oper);

   /// @name The constructors are protected
   ///@{
   SundialsSolver() : sundials_mem(NULL) { }
   SundialsSolver(void *mem) : sundials_mem(mem) { }
#ifdef MFEM_USE_MPI
   SundialsSolver(MPI_Comm comm_) : sundials_comm(comm_) { }
#endif
   ///@}

   /// Create an N_Vector.
   N_Vector CreateVector(long int length) const;

public:
   /// Access the underlying SUNDIALS object.
   void *SundialsMem() const { return sundials_mem; }

   /// Return the flag returned by the last call to a SUNDIALS function.
   int GetFlag() const { return flag; }
};

/// Wrapper for SUNDIALS' CVODE library -- Multi-step time integration.
/**
   - http://computation.llnl.gov/projects/sundials
   - http://computation.llnl.gov/sites/default/files/public/cv_guide.pdf

   @note All methods except Step() can be called before Init().
   To minimize uncertainty, we advise the user to adhere to the given
   interface, instead of making similar calls by the CVODE's
   internal CVodeMem object.
*/
class CVODESolver : public ODESolver, public SundialsSolver
{
public:
   /// Construct a serial CVODESolver, a wrapper for SUNDIALS' CVODE solver.
   /** @param[in] lmm   Specifies the linear multistep method, the options are
                        CV_ADAMS (explicit methods) or CV_BDF (implicit
                        methods).
       @param[in] iter  Specifies type of nonlinear solver iteration, the
                        options are CV_FUNCTIONAL (usually with CV_ADAMS) or
                        CV_NEWTON (usually with CV_BDF).
       For parameter desciption, see the CVodeCreate documentation (cvode.h). */
   CVODESolver(int lmm, int iter);

#ifdef MFEM_USE_MPI
   /// Construct a parallel CVODESolver, a wrapper for SUNDIALS' CVODE solver.
   /** @param[in] comm  The MPI communicator used to partition the ODE system.
       @param[in] lmm   Specifies the linear multistep method, the options are
                        CV_ADAMS (explicit methods) or CV_BDF (implicit
                        methods).
       @param[in] iter  Specifies type of nonlinear solver iteration, the
                        options are CV_FUNCTIONAL (usually with CV_ADAMS) or
                        CV_NEWTON (usually with CV_BDF).
       For parameter desciption, see the CVodeCreate documentation (cvode.h). */
   CVODESolver(MPI_Comm comm, int lmm, int iter);
#endif

   /// Set the scalar relative and scalar absolute tolerances.
   void SetSStolerances(double reltol, double abstol);

   /// Set a custom Jacobian system solver for the CV_NEWTON option usually used
   /// with implicit CV_BDF.
   void SetLinearSolver(SundialsODELinearSolver &ls_spec);

   /** @brief CVode supports two modes, specified by itask: CV_NORMAL (default)
       and CV_ONE_STEP. */
   /** In the CV_NORMAL mode, the solver steps until it reaches or passes
       tout = t + dt, where t and dt are specified in Step(), and then
       interpolates to obtain y(tout). In the CV_ONE_STEP mode, it takes one
       internal step and returns. */
   void SetStepMode(int itask);

   /// Set the maximum order of the linear multistep method.
   /** The default is 12 (CV_ADAMS) or 5 (CV_BDF).
       CVODE uses adaptive-order integration, based on the local truncation
       error. Use this if you know a priori that your system is such that
       higher order integration formulas are unstable.
       @note @a max_order can't be higher than the current maximum order. */
   void SetMaxOrder(int max_order);

   /// Set the maximum time step of the linear multistep method.
   void SetMaxStep(double dt_max)
   { flag = CVodeSetMaxStep(sundials_mem, dt_max); }

   /// Set the ODE right-hand-side operator.
   /** The start time of CVODE is initialized from the current time of @a f_.
       @note This method calls CVodeInit(). Some CVODE parameters can be set
       (using the handle returned by SundialsMem()) only after this call. */
   virtual void Init(TimeDependentOperator &f_);

   /// Use CVODE to integrate over [t, t + dt], with the specified step mode.
   /** Calls CVode(), which is the main driver of the CVODE package.
       @param[in,out] x  Solution vector to advance. On input/output x=x(t)
                         for t corresponding to the input/output value of t,
                         respectively.
       @param[in,out] t  Input: the starting time value. Output: the time value
                         of the solution output, as returned by CVode().
       @param[in,out] dt Input: desired time step. Output: the last incremental
                         time step used. */
   template <class TVector>
   void TStep(TVector &x, double &t, double &dt);

   virtual void Step(Vector &x, double &t, double &dt);
#ifdef MFEM_USE_OCCA
   virtual void Step(OccaVector &x, double &t, double &dt);
#endif

   /// Print CVODE statistics.
   void PrintInfo() const;

   /// Destroy the associated CVODE memory.
   virtual ~CVODESolver();
};

template <class TVector>
void CVODESolver::TStep(TVector &x, double &t, double &dt)
{
   CVodeMem mem = (CVodeMem) SundialsMem();

   MFEM_VERIFY(nv::GetLength(y) == static_cast<long int>(x.Size()), "");
   nv::SetVector(y, x);

   if (mem->cv_nst == 0)
   {
      // TOCONSIDER: Looks similar to sundials reinitialization.
      // Set default linear solver, if not already set.
      if (mem->cv_iter == CV_NEWTON && mem->cv_lsolve == NULL)
      {
         flag = CVSpgmr(sundials_mem, PREC_NONE, 0);
      }
      // Set the actual t0 and y0.
      mem->cv_tn = t;
      N_VScale(ONE, y, mem->cv_zn[0]);
   }

   double tout = t + dt;
   // The actual time integration.
   flag = CVode(sundials_mem, tout, y, &t, mem->cv_taskc);
   MFEM_ASSERT(flag >= 0, "CVode() failed!");

   // Return the last incremental step size.
   dt = mem->cv_hu;
}

/// Wrapper for SUNDIALS' ARKODE library -- Runge-Kutta time integration.
/**
  - http://computation.llnl.gov/projects/sundials
  - http://computation.llnl.gov/sites/default/files/public/ark_guide.pdf

   @note All methods except Step() can be called before Init().
   To minimize uncertainty, we advise the user to adhere to the given
   interface, instead of making similar calls by the ARKODE's
   internal ARKodeMem object.
*/
class ARKODESolver : public ODESolver, public SundialsSolver
{
protected:
   bool use_implicit;
   int irk_table, erk_table;

public:
   /// Types of ARKODE solvers.
   enum Type { EXPLICIT, IMPLICIT };

   /// Construct a serial ARKODESolver, a wrapper for SUNDIALS' ARKODE solver.
   /** @param[in] type  Specifies the #Type of ARKODE solver to construct. */
   ARKODESolver(Type type = EXPLICIT);

#ifdef MFEM_USE_MPI
   /// Construct a parallel ARKODESolver, a wrapper for SUNDIALS' ARKODE solver.
   /** @param[in] comm  The MPI communicator used to partition the ODE system.
       @param[in] type  Specifies the #Type of ARKODE solver to construct. */
   ARKODESolver(MPI_Comm comm, Type type = EXPLICIT);
#endif

   /// Specify the scalar relative and scalar absolute tolerances.
   void SetSStolerances(double reltol, double abstol);

   /// Set a custom Jacobian system solver for implicit methods.
   void SetLinearSolver(SundialsODELinearSolver &ls_spec);

   /** @brief ARKode supports two modes, specified by itask: ARK_NORMAL
       (default) and ARK_ONE_STEP. */
   /** In the ARK_NORMAL mode, the solver steps until it reaches or passes
       tout = t + dt, where t and dt are specified in Step(), and then
       interpolates to obtain y(tout). In the ARK_ONE_STEP mode, it takes one
       internal step and returns. */
   void SetStepMode(int itask);

   /// Chooses integration order for all explicit / implicit / IMEX methods.
   /** The default is 4, and the allowed ranges are: [2, 8] for explicit; [2, 5]
       for implicit; [3, 5] for IMEX. */
   void SetOrder(int order);

   /// Choose a specific Butcher table for implicit RK method.
   /** See the documentation for all possible options, stability regions, etc.
       For example, table_num = ARK548L2SA_DIRK_8_4_5 is 8-stage 5th order. */
   void SetIRKTableNum(int table_num);
   /// Choose a specific Butcher table for explicit RK method.
   /** See the documentation for all possible options, stability regions, etc.*/
   void SetERKTableNum(int table_num);

   /** @brief Use a fixed time step size, instead of performing any form of
       temporal adaptivity. */
   /** Use of this function is not recommended, since there is no assurance of
       the validity of the computed solutions. It is primarily provided for
       code-to-code verification testing purposes. */
   void SetFixedStep(double dt);

   /// Set the maximum time step of the Runge-Kutta method.
   void SetMaxStep(double dt_max)
   { flag = ARKodeSetMaxStep(sundials_mem, dt_max); }

   /// Set the ODE right-hand-side operator.
   /** The start time of ARKODE is initialized from the current time of @a f_.
       @note This method calls ARKodeInit(). Some ARKODE parameters can be set
       (using the handle returned by SundialsMem()) only after this call. */
   virtual void Init(TimeDependentOperator &f_);

   /// Use ARKODE to integrate over [t, t + dt], with the specified step mode.
   /** Calls ARKode(), which is the main driver of the ARKODE package.
       @param[in,out] x  Solution vector to advance. On input/output x=x(t)
                         for t corresponding to the input/output value of t,
                         respectively.
       @param[in,out] t  Input: the starting time value. Output: the time value
                         of the solution output, as returned by CVode().
       @param[in,out] dt Input: desired time step. Output: the last incremental
                         time step used. */
   template <class TVector>
   void TStep(TVector &x, double &t, double &dt);

   virtual void Step(Vector &x, double &t, double &dt);
#ifdef MFEM_USE_OCCA
   virtual void Step(OccaVector &x, double &t, double &dt);
#endif

   /// Print ARKODE statistics.
   void PrintInfo() const;

   /// Destroy the associated ARKODE memory.
   virtual ~ARKODESolver();
};

template <class TVector>
void ARKODESolver::TStep(TVector &x, double &t, double &dt)
{
   ARKodeMem mem = (ARKodeMem) SundialsMem();

   MFEM_VERIFY(nv::GetLength(y) == static_cast<long int>(x.Size()), "");
   nv::SetVector(y, x);

   if (mem->ark_nst == 0)
   {
      // Set default linear solver, if not already set.
      if (mem->ark_implicit && mem->ark_linit == NULL)
      {
         flag = ARKSpgmr(sundials_mem, PREC_NONE, 0);
      }
      // Set the actual t0 and y0.
      mem->ark_tn = t;
      mem->ark_tnew = t;

      N_VScale(ONE, y, mem->ark_ycur);
   }

   double tout = t + dt;
   // The actual time integration.
   flag = ARKode(sundials_mem, tout, y, &t, mem->ark_taskc);
   MFEM_ASSERT(flag >= 0, "ARKode() failed!");

   // Return the last incremental step size.
   dt = mem->ark_h;
}

/// Wrapper for SUNDIALS' KINSOL library -- Nonlinear solvers.
/**
   - http://computation.llnl.gov/projects/sundials
   - http://computation.llnl.gov/sites/default/files/public/kin_guide.pdf

   @note To minimize uncertainty, we advise the user to adhere to the given
   interface, instead of making similar calls by the KINSOL's
   internal KINMem object.
*/
class KinSolver : public NewtonSolver, public SundialsSolver
{
protected:
   bool use_oper_grad;
   mutable N_Vector y_scale, f_scale;
   const Operator *jacobian; // stores the result of oper->GetGradient()

   /// @name Auxiliary callback functions.
   ///@{
   // Computes the non-linear operator action F(u).
   // The real type of user_data is pointer to KinSolver.
   static int Mult(const N_Vector u, N_Vector fu, void *user_data);

   // Computes J(u)v. The real type of user_data is pointer to KinSolver.
   static int GradientMult(N_Vector v, N_Vector Jv, N_Vector u,
                           booleantype *new_u, void *user_data);

   static int LinSysSetup(KINMemRec *kin_mem);

   static int LinSysSolve(KINMemRec *kin_mem, N_Vector x, N_Vector b,
                          realtype *sJpnorm, realtype *sFdotJp);
   ///@}

public:
   /// Construct a serial KinSolver, a wrapper for SUNDIALS' KINSOL solver.
   /** @param[in] strategy   Specifies the nonlinear solver strategy:
                             KIN_NONE / KIN_LINESEARCH / KIN_PICARD / KIN_FP.
       @param[in] oper_grad  Specifies whether the solver should use its
                             Operator's GetGradient() method to compute the
                             Jacobian of the system. */
   KinSolver(int strategy, bool oper_grad = true);

#ifdef MFEM_USE_MPI
   /// Construct a parallel KinSolver, a wrapper for SUNDIALS' KINSOL solver.
   /** @param[in] comm       The MPI communicator used to partition the system.
       @param[in] strategy   Specifies the nonlinear solver strategy:
                             KIN_NONE / KIN_LINESEARCH / KIN_PICARD / KIN_FP.
       @param[in] oper_grad  Specifies whether the solver should use its
                             Operator's GetGradient() method to compute the
                             Jacobian of the system. */
   KinSolver(MPI_Comm comm, int strategy, bool oper_grad = true);
#endif

   /// Destroy the associated KINSOL memory.
   virtual ~KinSolver();

   /// Set the nonlinear Operator of the system. This method calls KINInit().
   virtual void SetOperator(const Operator &op);

   /// Set the linear solver for inverting the Jacobian.
   /** @note This function assumes that Operator::GetGradient(const Vector &)
             is implemented by the Operator specified by
             SetOperator(const Operator &). */
   virtual void SetSolver(Solver &solver);
   /// Equivalent to SetSolver(Solver).
   virtual void SetPreconditioner(Solver &solver) { SetSolver(solver); }

   /// Set KINSOL's scaled step tolerance.
   /** The default tolerance is U^(2/3), where U = machine unit roundoff. */
   void SetScaledStepTol(double sstol);
   /// Set KINSOL's functional norm tolerance.
   /** The default tolerance is U^(1/3), where U = machine unit roundoff.
        @note This function is equivalent to SetAbsTol(double). */
   void SetFuncNormTol(double ftol) { abs_tol = ftol; }

   /// Set maximum number of nonlinear iterations without a Jacobian update.
   /** The default is 10. */
   void SetMaxSetupCalls(int max_calls);

   /// Solve the nonlinear system F(x) = 0.
   /** Calls the other Mult(Vector&, Vector&, Vector&) const method with
       `x_scale = 1`. The values of 'fx_scale' are determined by comparing
       the chosen relative and functional norm (i.e. absolute) tolerances.
       @param[in]     b  Not used, KINSol always assumes zero RHS.
       @param[in,out] x  On input, initial guess, if @a #iterative_mode = true,
                         otherwise the initial guess is zero; on output, the
                         solution. */

   template <class TVector>
   void TMult(const TVector &b, TVector &x) const;

   virtual void Mult(const Vector &b, Vector &x) const;
#ifdef MFEM_USE_OCCA
   virtual void Mult(const OccaVector &b, OccaVector &x) const;
#endif

   /// Solve the nonlinear system F(x) = 0.
   /** Calls KINSol() to solve the nonlinear system. Before calling KINSol(),
       this functions uses the data members inherited from class IterativeSolver
       to set corresponding KINSOL options.
       @param[in,out] x        On input, initial guess, if @a #iterative_mode =
                               true, otherwise the initial guess is zero; on
                               output, the solution.
       @param[in]     x_scale  Elements of a diagonal scaling matrix D, s.t.
                               D*x has all elements roughly the same when
                               x is close to a solution.
       @param[in]    fx_scale  Elements of a diagonal scaling matrix E, s.t.
                               D*F(x) has all elements roughly the same when
                               x is not too close to a solution. */
   void Mult(Vector &x, const Vector &x_scale, const Vector &fx_scale) const;
#ifdef MFEM_USE_OCCA
   void Mult(OccaVector &x, const OccaVector &x_scale, const OccaVector &fx_scale) const;
#endif

  template <class TVector>
  void TMult(TVector &x, const TVector &x_scale, const TVector &fx_scale) const;

};

template <class TVector>
void KinSolver::TMult(const TVector &b, TVector &x) const
{
   KINMem mem = (KINMem) SundialsMem();

   // Uses c = 1, corresponding to x_scale.
   c = 1.0;

   if (!iterative_mode) { x = 0.0; }

   // For relative tolerance, r = 1 / |residual(x)|, corresponding to fx_scale.
   if (rel_tol > 0.0)
   {
      oper->Mult(x, r);

      // Note that KINSOL uses infinity norms.
      double norm;
#ifdef MFEM_USE_MPI
      if (Parallel())
      {
         double lnorm = r.Normlinf();
         MPI_Allreduce(&lnorm, &norm, 1, MPI_DOUBLE, MPI_MAX, NV_COMM_P(y));
      }
      else
 #endif
      {
         norm = r.Normlinf();
      }

      if (abs_tol > rel_tol * norm)
      {
         r = 1.0;
         mem->kin_fnormtol = abs_tol;
      }
      else
      {
         r =  1.0 / norm;
         mem->kin_fnormtol = rel_tol;
      }
   }
   else
   {
      mem->kin_fnormtol = abs_tol;
      r = 1.0;
   }
}

template <class TVector>
void KinSolver::TMult(TVector &x, const TVector &x_scale, const TVector &fx_scale) const
{
   KINMem mem = (KINMem) SundialsMem();

   flag = KINSetPrintLevel(sundials_mem, print_level);
   MFEM_ASSERT(flag >= 0, "KINSetPrintLevel() failed!");

   flag = KINSetNumMaxIters(sundials_mem, max_iter);
   MFEM_ASSERT(flag >= 0, "KINSetNumMaxIters() failed!");

   flag = KINSetScaledStepTol(sundials_mem, mem->kin_scsteptol);
   MFEM_ASSERT(flag >= 0, "KINSetScaledStepTol() failed!");

   flag = KINSetFuncNormTol(sundials_mem, mem->kin_fnormtol);
   MFEM_ASSERT(flag >= 0, "KINSetFuncNormTol() failed!");

   MFEM_VERIFY(nv::GetLength(y) == static_cast<long int>(x.Size()), "");
   nv::SetVector(y, x);
   nv::SetVector(y_scale, const_cast<TVector &>(x_scale));
   nv::SetVector(f_scale, const_cast<TVector &>(fx_scale));

   if (!iterative_mode) { x = 0.0; }

   flag = KINSol(sundials_mem, y, mem->kin_globalstrategy, y_scale, f_scale);

   converged  = (flag >= 0);
   final_iter = mem->kin_nni;
   final_norm = mem->kin_fnorm;
}


}  // namespace mfem

#endif // MFEM_USE_SUNDIALS

#endif // MFEM_SUNDIALS
