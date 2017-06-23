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

#include "sundials.hpp"

#ifdef MFEM_USE_SUNDIALS

#include "solvers.hpp"
#ifdef MFEM_USE_MPI
#include "hypre.hpp"
#endif

#include <nvector/nvector_serial.h>

#if defined(MFEM_USE_NVECTOR_CUDA) || defined(MFEM_USE_NVECTOR_OCCA)
#include <occa/modes/cuda.hpp>
#endif
#ifdef MFEM_USE_NVECTOR_CUDA
#include <nvector/nvector_cuda.h>
#include <nvector/cuda/Vector.hpp>
#endif

#ifdef MFEM_USE_MPI
#include <nvector/nvector_parallel.h>
#endif

// #include <cvode/cvode_impl.h>
// #include <cvode/cvode_spgmr.h>

// // This just hides a warning (to be removed after it's fixed in SUNDIALS).
// #ifdef MSG_TIME_INT
// #undef MSG_TIME_INT
// #endif

// #include <arkode/arkode_impl.h>
// #include <arkode/arkode_spgmr.h>

// #include <kinsol/kinsol_impl.h>
// #include <kinsol/kinsol_spgmr.h>

#ifdef MFEM_USE_NVECTOR_CUDA
typedef nvec::Vector<double, long int> SundialsCudaVector;
#endif

using namespace std;

namespace mfem
{

#ifdef MFEM_USE_NVECTOR_OCCA

// Operations for Occa NVector implementation
namespace ocs {

// Helper functions for one-shot get/set vector
static OccaVector& ExtractVector(N_Vector w)
{
   NVOCCAContent *content = (NVOCCAContent *) w->content;
   return *(content->vec);
}

static void SetVector(OccaVector &v, N_Vector w)
{
   NVOCCAContent *content = (NVOCCAContent *) w->content;
   content->vec = &v;
}

// Core operations
static N_Vector_ID nvgetvectorid(N_Vector w) { return SUNDIALS_NVEC_CUDA; }

static N_Vector nvcloneempty(N_Vector w)
{
   N_Vector v = new _generic_N_Vector;

   // Fill content
   v->content = new NVOCCAContent;
   NVOCCAContent *vcontent = (NVOCCAContent *) v->content;
   vcontent->ownVector = false;

   // Fill ops
   v->ops = new _generic_N_Vector_Ops;
   _generic_N_Vector_Ops *ops = v->ops;

   ops->nvgetvectorid     = w->ops->nvgetvectorid;
   ops->nvclone           = w->ops->nvclone;
   ops->nvcloneempty      = w->ops->nvcloneempty;
   ops->nvdestroy         = w->ops->nvdestroy;
   ops->nvspace           = w->ops->nvspace;
   ops->nvgetarraypointer = w->ops->nvgetarraypointer;
   ops->nvsetarraypointer = w->ops->nvsetarraypointer;
   ops->nvlinearsum       = w->ops->nvlinearsum;
   ops->nvconst           = w->ops->nvconst;
   ops->nvprod            = w->ops->nvprod;
   ops->nvdiv             = w->ops->nvdiv;
   ops->nvscale           = w->ops->nvscale;
   ops->nvabs             = w->ops->nvabs;
   ops->nvinv             = w->ops->nvinv;
   ops->nvaddconst        = w->ops->nvaddconst;
   ops->nvdotprod         = w->ops->nvdotprod;
   ops->nvmaxnorm         = w->ops->nvmaxnorm;
   ops->nvwrmsnormmask    = w->ops->nvwrmsnormmask;
   ops->nvwrmsnorm        = w->ops->nvwrmsnorm;
   ops->nvmin             = w->ops->nvmin;
   ops->nvwl2norm         = w->ops->nvwl2norm;
   ops->nvl1norm          = w->ops->nvl1norm;
   ops->nvcompare         = w->ops->nvcompare;
   ops->nvinvtest         = w->ops->nvinvtest;
   ops->nvconstrmask      = w->ops->nvconstrmask;
   ops->nvminquotient     = w->ops->nvminquotient;

   return v;
}

static N_Vector nvclone(N_Vector w)
{
   N_Vector v = nvcloneempty(w);
   v->content = w->content;

   return v;
}

static void nvdestroy(N_Vector w)
{
   NVOCCAContent *wcontent = (NVOCCAContent *) w->content;
   delete w->ops;
   if (wcontent->ownVector) delete wcontent->vec;
   delete wcontent;
   delete w;
}

static void nvspace(N_Vector v, long int *lrw, long int *liw)
{
   OccaVector &vec = ExtractVector(v);
   *lrw = vec.Size();
   *liw = 1;
}

static realtype* nvgetarraypointer(N_Vector v)
{
   OccaVector &vec = ExtractVector(v);
   return (realtype *) vec.GetData().ptr();
}

static void nvsetarraypointer(realtype *v_data, N_Vector v)
{
   OccaVector &vec = ExtractVector(v);
   const std::string mode = vec.GetData().getDevice().mode();

   if (mode == "CUDA")
   {
      occa::memory mem =
         occa::cuda::wrapMemory(occa::getDevice(), v_data, vec.Size() * sizeof(realtype *));
      vec.NewDataAndSize(mem, vec.Size());
   }
   else
   {
      mfem_error("Not yet implemented");
   }
}

static void nvlinearsum(realtype a, N_Vector x, realtype b, N_Vector y, N_Vector z)
{
   OccaVector &vx = ExtractVector(x);
   OccaVector &vy = ExtractVector(y);
   OccaVector &vz = ExtractVector(z);
   if (z == y)
   {
      vy.Set(a, vx);
   }
   else if (z == x)
   {
      vx.Set(b, vy);
   }
   else
   {
      vz.Set(a, vx);
      vz.Add(b, vy);
   }
}

static void nvconst(realtype c, N_Vector z)
{
   OccaVector &vz = ExtractVector(z);
   vz *= c;
}

static void nvprod(N_Vector x, N_Vector y, N_Vector z)
{
   OccaVector &vx = ExtractVector(x);
   OccaVector &vy = ExtractVector(y);
   OccaVector &vz = ExtractVector(z);
   vz = vx;
   vz *= vy;
}

static void nvdiv(N_Vector x, N_Vector y, N_Vector z)
{
   OccaVector &vx = ExtractVector(x);
   OccaVector &vy = ExtractVector(y);
   OccaVector &vz = ExtractVector(z);

   vz = vx;
   vz /= vy;
}

static void nvscale(realtype c, N_Vector x, N_Vector z)
{
   OccaVector &vx = ExtractVector(x);
   OccaVector &vz = ExtractVector(z);
   if (z == x)
   {
      vx *= c;
   }
   else
   {
      vz.Set(c, vx);
   }
}

static void nvabs(N_Vector x, N_Vector z)
{
   OccaVector &vx = ExtractVector(x);
   OccaVector &vz = ExtractVector(z);

   vz.NewDataAndSize(vx.GetData(), vx.Size());
   vz.Abs();
}

static void nvinv(N_Vector x, N_Vector z)
{
   OccaVector &vx = ExtractVector(x);
   OccaVector &vz = ExtractVector(z);

   static occa::kernelBuilder builder =
      makeCustomBuilder("vector_inv",
                        "v0[i] = 1.0 / v1[i];");

   occa::kernel kernel = builder.build(occa::getDevice());
   kernel((int) vz.Size(), vz.GetData(), vx.GetData());
}

static void nvaddconst(N_Vector x, realtype b, N_Vector z)
{
   OccaVector &vx = ExtractVector(x);
   OccaVector &vz = ExtractVector(z);

   vz.NewDataAndSize(vx.GetData(), vx.Size());
   vz += b;
}

static realtype nvdotprod(N_Vector x, N_Vector y)
{
   OccaVector &vx = ExtractVector(x);
   OccaVector &vy = ExtractVector(y);

   return vx * vy;
}

static realtype nvmaxnorm(N_Vector x)
{
   OccaVector &vx = ExtractVector(x);
   return occa::linalg::lInfNorm<double,double>(vx.GetData());
}

static realtype nvwrmsnorm(N_Vector x, N_Vector w)
{
   mfem_error("Not yet implemented");
}

static realtype nvwrmsnormmask(N_Vector, N_Vector, N_Vector)
{
   mfem_error("Not yet implemented");
}

static realtype nvmin(N_Vector x)
{
   OccaVector &vx = ExtractVector(x);
   return occa::linalg::min<double,double>(vx.GetData());
}

static realtype nvwl2norm(N_Vector, N_Vector)
{
   mfem_error("Not yet implemented");
}

static realtype nvl1norm(N_Vector x)
{
   OccaVector &vx = ExtractVector(x);
   return occa::linalg::l1Norm<double,double>(vx.GetData());
}

static void nvcompare(realtype, N_Vector, N_Vector)
{
   mfem_error("Not yet implemented");
}

static booleantype nvinvtest(N_Vector, N_Vector)
{
   mfem_error("Not yet implemented");
}

static booleantype nvconstrmask(N_Vector, N_Vector, N_Vector)
{
   mfem_error("Not yet implemented");
}

static realtype nvminquotient(N_Vector, N_Vector)
{
   mfem_error("Not yet implemented");
}

} // namespace ocs

static N_Vector ConstructOccaNVector()
{
   N_Vector y = new _generic_N_Vector;
   y->content = new NVOCCAContent;
   y->ops     = new _generic_N_Vector_Ops;

   // Fill operations
   _generic_N_Vector_Ops *ops = y->ops;
   ops->nvgetvectorid     = ocs::nvgetvectorid;
   ops->nvclone           = ocs::nvclone;
   ops->nvcloneempty      = ocs::nvcloneempty;
   ops->nvdestroy         = ocs::nvdestroy;
   ops->nvspace           = ocs::nvspace;
   ops->nvgetarraypointer = ocs::nvgetarraypointer;
   ops->nvsetarraypointer = ocs::nvsetarraypointer;
   ops->nvlinearsum       = ocs::nvlinearsum;
   ops->nvconst           = ocs::nvconst;
   ops->nvprod            = ocs::nvprod;
   ops->nvdiv             = ocs::nvdiv;
   ops->nvscale           = ocs::nvscale;
   ops->nvabs             = ocs::nvabs;
   ops->nvinv             = ocs::nvinv;
   ops->nvaddconst        = ocs::nvaddconst;
   ops->nvdotprod         = ocs::nvdotprod;
   ops->nvmaxnorm         = ocs::nvmaxnorm;
   ops->nvwrmsnormmask    = ocs::nvwrmsnormmask;
   ops->nvwrmsnorm        = ocs::nvwrmsnorm;
   ops->nvmin             = ocs::nvmin;
   ops->nvwl2norm         = ocs::nvwl2norm;
   ops->nvl1norm          = ocs::nvl1norm;
   ops->nvcompare         = ocs::nvcompare;
   ops->nvinvtest         = ocs::nvinvtest;
   ops->nvconstrmask      = ocs::nvconstrmask;
   ops->nvminquotient     = ocs::nvminquotient;

   return y;
}

static N_Vector MakeOccaNVector(OccaVector &v)
{
   N_Vector y = ConstructOccaNVector();

   // Fill content
   NVOCCAContent *ycontent = (NVOCCAContent *) y->content;
   ycontent->vec = &v;
   ycontent->ownVector = false;
#ifdef MFEM_USE_MPI
   ycontent->comm = sundials_comm;
   mfem_error("Not yet supported");
#endif

   return y;
}

static N_Vector NewOccaNVector(long int length)
{
   N_Vector y = ConstructOccaNVector();

   // Fill content
   NVOCCAContent *ycontent = (NVOCCAContent *) y->content;
   ycontent->vec = new OccaVector(length);
   ycontent->ownVector = true;
#ifdef MFEM_USE_MPI
   ycontent->comm = sundials_comm;
   mfem_error("Not yet supported");
#endif

   return y;
}

#endif // ifdef MFEM_USE_NVECTOR_OCCA


double SundialsODELinearSolver::GetTimeStep(void *sundials_mem)
{
   return (type == CVODE) ?
          ((CVodeMem)sundials_mem)->cv_gamma :
          ((ARKodeMem)sundials_mem)->ark_gamma;
}

TimeDependentOperator *
SundialsODELinearSolver::GetTimeDependentOperator(void *sundials_mem)
{
   void *user_data = (type == CVODE) ?
                     ((CVodeMem)sundials_mem)->cv_user_data :
                     ((ARKodeMem)sundials_mem)->ark_user_data;
   return (TimeDependentOperator *)user_data;
}

static inline SundialsODELinearSolver *to_solver(void *ptr)
{
   return static_cast<SundialsODELinearSolver *>(ptr);
}

static int cvLinSysInit(CVodeMem cv_mem)
{
   return to_solver(cv_mem->cv_lmem)->InitSystem(cv_mem);
}

static int cvLinSysSetup(CVodeMem cv_mem, int convfail,
                         N_Vector ypred, N_Vector fpred, booleantype *jcurPtr,
                         N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3)
{
   Vector yp(ypred), fp(fpred), vt1(vtemp1), vt2(vtemp2), vt3(vtemp3);
   return to_solver(cv_mem->cv_lmem)->SetupSystem(cv_mem, convfail, yp, fp,
                                                  *jcurPtr, vt1, vt2, vt3);
}

static int cvLinSysSolve(CVodeMem cv_mem, N_Vector b, N_Vector weight,
                         N_Vector ycur, N_Vector fcur)
{
   Vector bb(b), w(weight), yc(ycur), fc(fcur);
   return to_solver(cv_mem->cv_lmem)->SolveSystem(cv_mem, bb, w, yc, fc);
}

static int cvLinSysFree(CVodeMem cv_mem)
{
   return to_solver(cv_mem->cv_lmem)->FreeSystem(cv_mem);
}

static int arkLinSysInit(ARKodeMem ark_mem)
{
   return to_solver(ark_mem->ark_lmem)->InitSystem(ark_mem);
}

static int arkLinSysSetup(ARKodeMem ark_mem, int convfail,
                          N_Vector ypred, N_Vector fpred, booleantype *jcurPtr,
                          N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3)
{
   Vector yp(ypred), fp(fpred), vt1(vtemp1), vt2(vtemp2), vt3(vtemp3);
   return to_solver(ark_mem->ark_lmem)->SetupSystem(ark_mem, convfail, yp, fp,
                                                    *jcurPtr, vt1, vt2, vt3);
}

static int arkLinSysSolve(ARKodeMem ark_mem, N_Vector b, N_Vector weight,
                          N_Vector ycur, N_Vector fcur)
{
   Vector bb(b), w(weight), yc(ycur), fc(fcur);
   return to_solver(ark_mem->ark_lmem)->SolveSystem(ark_mem, bb, w, yc, fc);
}

static int arkLinSysFree(ARKodeMem ark_mem)
{
   return to_solver(ark_mem->ark_lmem)->FreeSystem(ark_mem);
}

// TODO: The vectorType hack can be removed when merging
int SundialsSolver::vectorType;
const double SundialsSolver::default_rel_tol = 1e-4;
const double SundialsSolver::default_abs_tol = 1e-9;

// static method
int SundialsSolver::ODEMult(realtype t, const N_Vector y,
                            N_Vector ydot, void *td_oper)
{
   TimeDependentOperator *f = static_cast<TimeDependentOperator *>(td_oper);

   if (vectorType == 1)
   {
      const Vector mfem_y(y);
      Vector mfem_ydot(ydot);
      // Compute y' = f(t, y).

      f->SetTime(t);
      f->Mult(mfem_y, mfem_ydot);
   }
   else
   {
      const OccaVector mfem_y(y);
      OccaVector mfem_ydot(ydot);

      f->SetTime(t);
      f->Mult(mfem_y, mfem_ydot);
   }

   return 0;
}

N_Vector SundialsSolver::CreateVector(long int length) const
{
   N_Vector nv;

   /* Temporarily convoluted check */
#if defined(MFEM_USE_OCCA)
   const std::string mode = occa::getDevice().mode();
#else
   const std::string mode = "Serial";
#endif

   if ((mode == "Serial") || (mode == "OpenMP"))
   {
#ifdef MFEM_USE_MPI
      long int global_length;
      MPI_Allreduce(&length, &global_length, 1, MPI_LONG, MPI_SUM, sundials_comm);
      nv = N_VNew_Parallel(sundials_comm, length, global_length);
#else
      nv = N_VNew_Serial(length);
#endif
   }
   else
   {
#if defined(MFEM_USE_NVECTOR_CUDA)
      nv = N_VNew_Cuda(length);
#elif defined(MFEM_USE_NVECTOR_OCCA)
      nv = NewOccaNVector(length);
#else
      mfem_error("Type not supported in SundialsSolver::CreateVector()");
#endif
   }

   return nv;
}

static inline CVodeMem Mem(const CVODESolver *self)
{
   return CVodeMem(self->SundialsMem());
}

namespace nv
{

void Destroy(N_Vector &nv)
{
   N_Vector_ID nvid = N_VGetVectorID(nv);

   if (nvid == SUNDIALS_NVEC_SERIAL)
   {
      N_VectorContent_Serial content = (N_VectorContent_Serial) nv->content;
      long int length = content->length;
      nv = N_VNewEmpty_Serial(length);
   }
#ifdef MFEM_USE_MPI
   else if (nvid == SUNDIALS_NVEC_PARALLEL)
   {
      N_VectorContent_Parallel content = (N_VectorContent_Parallel) nv->content;
      long int local_length = content->local_length;
      long int global_length = content->global_length;
      MPI_Comm comm = content->comm;
      N_VDestroy(nv);
      nv = N_VNewEmpty_Parallel(comm, local_length, global_length);
   }
#endif
   else
   {
#if defined(MFEM_USE_NVECTOR_CUDA)
      // do nothing
#elif defined(MFEM_USE_NVECTOR_OCCA)
      // delete the OccaVector
      NVOCCAContent *content = (NVOCCAContent *) nv->content;
      if (content->ownVector) delete content->vec;
#else
      mfem_error("Type not supported in nv::Destroy()");
#endif
   }
}

void SetVector(const N_Vector &nv, Vector &v)
{
   N_Vector_ID nvid = N_VGetVectorID(nv);

   if (nvid == SUNDIALS_NVEC_SERIAL)
   {
     N_VectorContent_Serial content = (N_VectorContent_Serial) nv->content;
     if (content->own_data) mfem_error("Need to nv::Destroy() data first!");
     content->data = v.GetData();
     content->own_data = false;
   }
#ifdef MFEM_USE_MPI
   else if (nvid == SUNDIALS_NVEC_PARALLEL)
   {
     N_VectorContent_Parallel content = (N_VectorContent_Parallel) nv->content;
      if (content->own_data) mfem_error("Need to nv::Destroy() data first!");
      content->data = v.GetData();
      content->own_data = false;
   }
#endif
   else
   {
#if defined(MFEM_USE_NVECTOR_CUDA)
      SundialsCudaVector *content = (SundialsCudaVector *) nv->content;
      content->setFromHost(v.GetData());
#elif defined(MFEM_USE_NVECTOR_OCCA)
      NVOCCAContent *content = (NVOCCAContent *) nv->content;
      content->vec->NewDataAndSize(v.GetData(), v.Size());
#else
      mfem_error("Type not supported in nv::SetVector()");
#endif
   }
}

#ifdef MFEM_USE_OCCA
void SetVector(const N_Vector &nv, OccaVector &v)
{
   N_Vector_ID nvid = N_VGetVectorID(nv);
   const std::string mode = v.GetDevice().mode();

   if (nvid == SUNDIALS_NVEC_SERIAL)
   {
     if (mode == "CUDA") mfem_error("OccaVector type not supported");
     N_VectorContent_Serial content = (N_VectorContent_Serial) nv->content;
     if (content->own_data) mfem_error("Need to nv::Destroy() data first!");
     content->data = (realtype *) v.GetData().ptr();
     content->own_data = false;
   }
#ifdef MFEM_USE_MPI
   else if (nvid == SUNDIALS_NVEC_PARALLEL)
   {
     N_VectorContent_Parallel content = (N_VectorContent_Parallel) nv->content;
     if (content->own_data) mfem_error("Need to nv::Destroy() data first!");
     content->data = (realtype *) v.GetData().ptr();
     content->own_data = false;
   }
#endif
   else
   {
#if defined(MFEM_USE_NVECTOR_CUDA)
      SundialsCudaVector *content = (SundialsCudaVector *) nv->content;
      content->setFromDevice((double *) v.GetData().ptr());
#elif defined(MFEM_USE_NVECTOR_OCCA)
      NVOCCAContent *content = (NVOCCAContent *) nv->content;
      content->vec = &v;
#else
      mfem_error("Type not supported in nv::SetVector()");
#endif
   }
}
#endif

long int GetLength(const N_Vector &nv)
{
   N_Vector_ID nvid = N_VGetVectorID(nv);
   long int length = -1;

   if (nvid == SUNDIALS_NVEC_SERIAL)
   {
      length = N_VGetLength_Serial(nv);
   }
#ifdef MFEM_USE_MPI
   else if (nvid == SUNDIALS_NVEC_PARALLEL)
   {
      length = N_VGetLocalLength_Parallel(nv);
   }
#endif
   else
   {
#if defined(MFEM_USE_NVECTOR_CUDA)
      SundialsCudaVector *content = (SundialsCudaVector *) nv->content;
      length = content->size();
#elif defined(MFEM_USE_NVECTOR_OCCA)
      NVOCCAContent *content = (NVOCCAContent *) nv->content;
      length = content->vec->Size();
#else
      mfem_error("Type not supported in nv::GetLength()");
#endif
   }

   return length;
}

} // namespace nv


CVODESolver::CVODESolver(int lmm, int iter) : SundialsSolver()
{
#if defined(MFEM_USE_OCCA) && defined(MFEM_USE_NVECTOR_CUDA)
   // pass the context to sundials
   nvec::setCudaContext(occa::cuda::getContext(occa::getDevice()));
#endif

   // Create the solver memory.
   sundials_mem = CVodeCreate(lmm, iter);
   MFEM_ASSERT(sundials_mem, "error in CVodeCreate()");

   SetStepMode(CV_NORMAL);

   // Replace the zero defaults with some positive numbers.
   SetSStolerances(default_rel_tol, default_abs_tol);

   flag = CV_SUCCESS;
}

#ifdef MFEM_USE_MPI

CVODESolver::CVODESolver(MPI_Comm comm, int lmm, int iter) : SundialsSolver(comm)
{
   // Create the solver memory.
   sundials_mem = CVodeCreate(lmm, iter);
   MFEM_ASSERT(sundials_mem, "error in CVodeCreate()");

   SetStepMode(CV_NORMAL);

   // Replace the zero defaults with some positive numbers.
   SetSStolerances(default_rel_tol, default_abs_tol);

   flag = CV_SUCCESS;
}

#endif // MFEM_USE_MPI

void CVODESolver::SetSStolerances(double reltol, double abstol)
{
   CVodeMem mem = Mem(this);
   // For now store the values in mem:
   mem->cv_reltol = reltol;
   mem->cv_Sabstol = abstol;
   // The call to CVodeSStolerances() is done after CVodeInit() in Init().
}

void CVODESolver::SetLinearSolver(SundialsODELinearSolver &ls_spec)
{
   CVodeMem mem = Mem(this);
   MFEM_ASSERT(mem->cv_iter == CV_NEWTON,
               "The function is applicable only to CV_NEWTON iteration type.");

   if (mem->cv_lfree != NULL) { (mem->cv_lfree)(mem); }

   // Set the linear solver function fields in mem.
   // Note that {linit,lsetup,lfree} can be NULL.
   mem->cv_linit  = cvLinSysInit;
   mem->cv_lsetup = cvLinSysSetup;
   mem->cv_lsolve = cvLinSysSolve;
   mem->cv_lfree  = cvLinSysFree;
   mem->cv_lmem   = &ls_spec;
   mem->cv_setupNonNull = TRUE;
   ls_spec.type = SundialsODELinearSolver::CVODE;
}

void CVODESolver::SetStepMode(int itask)
{
   Mem(this)->cv_taskc = itask;
}

void CVODESolver::SetMaxOrder(int max_order)
{
   flag = CVodeSetMaxOrd(sundials_mem, max_order);
   if (flag == CV_ILL_INPUT)
   {
      MFEM_WARNING("CVodeSetMaxOrd() did not change the maximum order!");
   }
}

// Has to copy all fields that can be set by the MFEM interface !!
static inline void cvCopyInit(CVodeMem src, CVodeMem dest)
{
   dest->cv_lmm  = src->cv_lmm;
   dest->cv_iter = src->cv_iter;

   dest->cv_linit  = src->cv_linit;
   dest->cv_lsetup = src->cv_lsetup;
   dest->cv_lsolve = src->cv_lsolve;
   dest->cv_lfree  = src->cv_lfree;
   dest->cv_lmem   = src->cv_lmem;
   dest->cv_setupNonNull = src->cv_setupNonNull;

   dest->cv_reltol  = src->cv_reltol;
   dest->cv_Sabstol = src->cv_Sabstol;

   dest->cv_taskc = src->cv_taskc;
   dest->cv_qmax = src->cv_qmax;

   // Do not copy cv_hmax_inv, it is not overwritten by CVodeInit.
}

void CVODESolver::Init(TimeDependentOperator &f_)
{
   CVodeMem mem = Mem(this);
   CVodeMemRec backup;

   if (mem->cv_MallocDone == TRUE)
   {
      // TODO: preserve more options.
      cvCopyInit(mem, &backup);
      CVodeFree(&sundials_mem);
      sundials_mem = CVodeCreate(backup.cv_lmm, backup.cv_iter);
      MFEM_ASSERT(sundials_mem, "error in CVodeCreate()");
      cvCopyInit(&backup, mem);
   }

   ODESolver::Init(f_);

   // Have SundialsSolver create the N_Vector y.
   y = CreateVector(f_.Height());

   // Call CVodeInit().
   cvCopyInit(mem, &backup);
   flag = CVodeInit(mem, ODEMult, f_.GetTime(), y);
   MFEM_ASSERT(flag >= 0, "CVodeInit() failed!");
   cvCopyInit(&backup, mem);

   // Delete the allocated data in y.
   nv::Destroy(y);

   // The TimeDependentOperator pointer, f, will be the user-defined data.
   flag = CVodeSetUserData(sundials_mem, f);
   MFEM_ASSERT(flag >= 0, "CVodeSetUserData() failed!");

   flag = CVodeSStolerances(mem, mem->cv_reltol, mem->cv_Sabstol);
   MFEM_ASSERT(flag >= 0, "CVodeSStolerances() failed!");
}

void CVODESolver::Step(Vector &x, double &t, double &dt)
{
  SundialsSolver::vectorType = 1;
  TStep(x, t, dt);
}

#ifdef MFEM_USE_OCCA
void CVODESolver::Step(OccaVector &x, double &t, double &dt)
{
  SundialsSolver::vectorType = 2;
  TStep(x, t, dt);
}
#endif

void CVODESolver::PrintInfo() const
{
   CVodeMem mem = Mem(this);

   cout <<
        "CVODE:\n  "
        "num steps: " << mem->cv_nst << ", "
        "num evals: " << mem->cv_nfe << ", "
        "num lin setups: " << mem->cv_nsetups << ", "
        "num nonlin sol iters: " << mem->cv_nni << "\n  "
        "last order: " << mem->cv_qu << ", "
        "next order: " << mem->cv_next_q << ", "
        "last dt: " << mem->cv_hu << ", "
        "next dt: " << mem->cv_next_h
        << endl;
}

CVODESolver::~CVODESolver()
{
   N_VDestroy(y);
   CVodeFree(&sundials_mem);
}

static inline ARKodeMem Mem(const ARKODESolver *self)
{
   return ARKodeMem(self->SundialsMem());
}

ARKODESolver::ARKODESolver(Type type)
   : SundialsSolver(), use_implicit(type == IMPLICIT), irk_table(-1), erk_table(-1)
{
   // Create the solver memory.
   sundials_mem = ARKodeCreate();
   MFEM_ASSERT(sundials_mem, "error in ARKodeCreate()");

   SetStepMode(ARK_NORMAL);

   // Replace the zero defaults with some positive numbers.
   SetSStolerances(default_rel_tol, default_abs_tol);

   flag = ARK_SUCCESS;
}

#ifdef MFEM_USE_MPI
ARKODESolver::ARKODESolver(MPI_Comm comm, Type type)
   : SundialsSolver(comm), use_implicit(type == IMPLICIT), irk_table(-1), erk_table(-1)
{
   // Create the solver memory.
   sundials_mem = ARKodeCreate();
   MFEM_ASSERT(sundials_mem, "error in ARKodeCreate()");

   SetStepMode(ARK_NORMAL);

   // Replace the zero defaults with some positive numbers.
   SetSStolerances(default_rel_tol, default_abs_tol);

   flag = ARK_SUCCESS;
}
#endif

void ARKODESolver::SetSStolerances(double reltol, double abstol)
{
   ARKodeMem mem = Mem(this);
   // For now store the values in mem:
   mem->ark_reltol  = reltol;
   mem->ark_Sabstol = abstol;
   // The call to ARKodeSStolerances() is done after ARKodeInit() in Init().
}

void ARKODESolver::SetLinearSolver(SundialsODELinearSolver &ls_spec)
{
   ARKodeMem mem = Mem(this);
   MFEM_VERIFY(use_implicit,
               "The function is applicable only to implicit time integration.");

   if (mem->ark_lfree != NULL) { mem->ark_lfree(mem); }

   // Tell ARKODE that the Jacobian inversion is custom.
   mem->ark_lsolve_type = 4;
   // Set the linear solver function fields in mem.
   // Note that {linit,lsetup,lfree} can be NULL.
   mem->ark_linit  = arkLinSysInit;
   mem->ark_lsetup = arkLinSysSetup;
   mem->ark_lsolve = arkLinSysSolve;
   mem->ark_lfree  = arkLinSysFree;
   mem->ark_lmem   = &ls_spec;
   mem->ark_setupNonNull = TRUE;
   ls_spec.type = SundialsODELinearSolver::ARKODE;
}

void ARKODESolver::SetStepMode(int itask)
{
   Mem(this)->ark_taskc = itask;
}

void ARKODESolver::SetOrder(int order)
{
   ARKodeMem mem = Mem(this);
   // For now store the values in mem:
   mem->ark_q = order;
   // The call to ARKodeSetOrder() is done after ARKodeInit() in Init().
}

void ARKODESolver::SetIRKTableNum(int table_num)
{
   // The call to ARKodeSetIRKTableNum() is done after ARKodeInit() in Init().
   irk_table = table_num;
}

void ARKODESolver::SetERKTableNum(int table_num)
{
   // The call to ARKodeSetERKTableNum() is done after ARKodeInit() in Init().
   erk_table = table_num;
}

void ARKODESolver::SetFixedStep(double dt)
{
   flag = ARKodeSetFixedStep(sundials_mem, dt);
   MFEM_ASSERT(flag >= 0, "ARKodeSetFixedStep() failed!");
}

// Copy fields that can be set by the MFEM interface.
static inline void arkCopyInit(ARKodeMem src, ARKodeMem dest)
{
   dest->ark_lsolve_type  = src->ark_lsolve_type;
   dest->ark_linit        = src->ark_linit;
   dest->ark_lsetup       = src->ark_lsetup;
   dest->ark_lsolve       = src->ark_lsolve;
   dest->ark_lfree        = src->ark_lfree;
   dest->ark_lmem         = src->ark_lmem;
   dest->ark_setupNonNull = src->ark_setupNonNull;

   dest->ark_reltol  = src->ark_reltol;
   dest->ark_Sabstol = src->ark_Sabstol;

   dest->ark_taskc     = src->ark_taskc;
   dest->ark_q         = src->ark_q;
   dest->ark_fixedstep = src->ark_fixedstep;
   dest->ark_hin       = src->ark_hin;
}

void ARKODESolver::Init(TimeDependentOperator &f_)
{
   ARKodeMem mem = Mem(this);
   ARKodeMemRec backup;

   // Check if ARKodeInit() has already been called.
   if (mem->ark_MallocDone == TRUE)
   {
      // TODO: preserve more options.
      arkCopyInit(mem, &backup);
      ARKodeFree(&sundials_mem);
      sundials_mem = ARKodeCreate();
      MFEM_ASSERT(sundials_mem, "Error in ARKodeCreate()!");
      arkCopyInit(&backup, mem);
   }

   ODESolver::Init(f_);

   // Have SundialsSolver create the N_Vector y.
   y = CreateVector(f_.Height());

   // Call ARKodeInit().
   arkCopyInit(mem, &backup);
   double t = f_.GetTime();
   // TODO: IMEX interface and example.
   flag = (use_implicit) ?
          ARKodeInit(sundials_mem, NULL, ODEMult, t, y) :
          ARKodeInit(sundials_mem, ODEMult, NULL, t, y);
   MFEM_ASSERT(flag >= 0, "CVodeInit() failed!");
   arkCopyInit(&backup, mem);

   // Delete the allocated data in y.
   nv::Destroy(y);

   // The TimeDependentOperator pointer, f, will be the user-defined data.
   flag = ARKodeSetUserData(sundials_mem, f);
   MFEM_ASSERT(flag >= 0, "ARKodeSetUserData() failed!");

   flag = ARKodeSStolerances(mem, mem->ark_reltol, mem->ark_Sabstol);
   MFEM_ASSERT(flag >= 0, "CVodeSStolerances() failed!");

   flag = ARKodeSetOrder(sundials_mem, mem->ark_q);
   MFEM_ASSERT(flag >= 0, "ARKodeSetOrder() failed!");

   if (irk_table >= 0)
   {
      flag = ARKodeSetIRKTableNum(sundials_mem, irk_table);
      MFEM_ASSERT(flag >= 0, "ARKodeSetIRKTableNum() failed!");
   }
   if (erk_table >= 0)
   {
      flag = ARKodeSetERKTableNum(sundials_mem, erk_table);
      MFEM_ASSERT(flag >= 0, "ARKodeSetERKTableNum() failed!");
   }
}

void ARKODESolver::Step(Vector &x, double &t, double &dt)
{
  vectorType = 1;
  TStep(x, t, dt);
}

#ifdef MFEM_USE_OCCA
void ARKODESolver::Step(OccaVector &x, double &t, double &dt)
{
  vectorType = 2;
  TStep(x, t, dt);
}
#endif

void ARKODESolver::PrintInfo() const
{
   ARKodeMem mem = Mem(this);

   cout <<
        "ARKODE:\n  "
        "num steps: " << mem->ark_nst << ", "
        "num evals: " << mem->ark_nfe << ", "
        "num lin setups: " << mem->ark_nsetups << ", "
        "num nonlin sol iters: " << mem->ark_nni << "\n  "
        "method order: " << mem->ark_q << ", "
        "last dt: " << mem->ark_h << ", "
        "next dt: " << mem->ark_next_h
        << endl;
}

ARKODESolver::~ARKODESolver()
{
   N_VDestroy(y);
   ARKodeFree(&sundials_mem);
}


static inline KINMem Mem(const KinSolver *self)
{
   return KINMem(self->SundialsMem());
}

// static method
int KinSolver::Mult(const N_Vector u, N_Vector fu, void *user_data)
{
#ifdef MFEM_USE_OCCA
   const OccaVector mfem_u(u);
   OccaVector mfem_fu(fu);
#else
   const Vector mfem_u(u);
   Vector mfem_fu(fu);
#endif

   // Computes the non-linear action F(u).
   static_cast<KinSolver*>(user_data)->oper->Mult(mfem_u, mfem_fu);
   return 0;
}

// static method
int KinSolver::GradientMult(N_Vector v, N_Vector Jv, N_Vector u,
                            booleantype *new_u, void *user_data)
{
#ifdef MFEM_USE_OCCA
   const OccaVector mfem_v(v);
   OccaVector mfem_Jv(Jv);
#else
   const Vector mfem_v(v);
   Vector mfem_Jv(Jv);
#endif

   KinSolver *self = static_cast<KinSolver*>(user_data);
   if (*new_u)
   {
      const Vector mfem_u(u);
      self->jacobian = &self->oper->GetGradient(mfem_u);
      *new_u = FALSE;
   }
   self->jacobian->Mult(mfem_v, mfem_Jv);
   return 0;
}

// static method
int KinSolver::LinSysSetup(KINMemRec *kin_mem)
{
   const Vector u(kin_mem->kin_uu);

   KinSolver *self = static_cast<KinSolver*>(kin_mem->kin_lmem);

   self->jacobian = &self->oper->GetGradient(u);
   self->prec->SetOperator(*self->jacobian);

   return KIN_SUCCESS;
}

// static method
int KinSolver::LinSysSolve(KINMemRec *kin_mem, N_Vector x, N_Vector b,
                           realtype *sJpnorm, realtype *sFdotJp)
{
#ifdef MFEM_USE_OCCA
   OccaVector mx(x), mb(b);
#else
   Vector mx(x), mb(b);
#endif
   KinSolver *self = static_cast<KinSolver*>(kin_mem->kin_lmem);

   // Solve for mx = [J(u)]^{-1} mb, maybe approximately.
   self->prec->Mult(mb, mx);

   // Compute required norms.
   if ( (kin_mem->kin_globalstrategy == KIN_LINESEARCH) ||
        (kin_mem->kin_globalstrategy != KIN_FP &&
         kin_mem->kin_etaflag == KIN_ETACHOICE1) )
   {
      // mb = J(u) mx - if the solve above was "exact", is this necessary?
      self->jacobian->Mult(mx, mb);

      *sJpnorm = N_VWL2Norm(b, kin_mem->kin_fscale);
      N_VProd(b, kin_mem->kin_fscale, b);
      N_VProd(b, kin_mem->kin_fscale, b);
      *sFdotJp = N_VDotProd(kin_mem->kin_fval, b);
      // Increment counters?
   }

   return KIN_SUCCESS;
}

KinSolver::KinSolver(int strategy, bool oper_grad)
   : SundialsSolver(), use_oper_grad(oper_grad), jacobian(NULL)
{
   sundials_mem = KINCreate();
   MFEM_ASSERT(sundials_mem, "Error in KINCreate().");

   Mem(this)->kin_globalstrategy = strategy;
   // Default abs_tol, print_level.
   abs_tol = Mem(this)->kin_fnormtol;
   print_level = 0;

   flag = KIN_SUCCESS;
}

#ifdef MFEM_USE_MPI

KinSolver::KinSolver(MPI_Comm comm, int strategy, bool oper_grad)
   : SundialsSolver(comm), use_oper_grad(oper_grad), jacobian(NULL)
{
   sundials_mem = KINCreate();
   MFEM_ASSERT(sundials_mem, "Error in KINCreate().");

   Mem(this)->kin_globalstrategy = strategy;
   // Default abs_tol, print_level.
   abs_tol = Mem(this)->kin_fnormtol;
   print_level = 0;

   flag = KIN_SUCCESS;
}

#endif

// Copy fields that can be set by the MFEM interface.
static inline void kinCopyInit(KINMem src, KINMem dest)
{
   dest->kin_linit        = src->kin_linit;
   dest->kin_lsetup       = src->kin_lsetup;
   dest->kin_lsolve       = src->kin_lsolve;
   dest->kin_lfree        = src->kin_lfree;
   dest->kin_lmem         = src->kin_lmem;
   dest->kin_setupNonNull = src->kin_setupNonNull;
   dest->kin_msbset       = src->kin_msbset;

   dest->kin_globalstrategy = src->kin_globalstrategy;
   dest->kin_printfl        = src->kin_printfl;
   dest->kin_mxiter         = src->kin_mxiter;
   dest->kin_scsteptol      = src->kin_scsteptol;
   dest->kin_fnormtol       = src->kin_fnormtol;
}

void KinSolver::SetOperator(const Operator &op)
{
   KINMem mem = Mem(this);
   KINMemRec backup;

   // Check if SetOperator() has already been called.
   if (mem->kin_MallocDone == TRUE)
   {
      // TODO: preserve more options.
      kinCopyInit(mem, &backup);
      KINFree(&sundials_mem);
      sundials_mem = KINCreate();
      MFEM_ASSERT(sundials_mem, "Error in KinCreate()!");
      kinCopyInit(&backup, mem);
   }

   NewtonSolver::SetOperator(op);
   jacobian = NULL;

   // Set actual size and data in the N_Vector y.
   y = CreateVector(height);
   y_scale = CreateVector(height);
   f_scale = CreateVector(height);

   kinCopyInit(mem, &backup);
   flag = KINInit(sundials_mem, KinSolver::Mult, y);
   // Initialization of kin_pp; otherwise, for a custom Jacobian inversion,
   // the first time we enter the linear solve, we will get uninitialized
   // initial guess (matters when iterative_mode = true).
   N_VConst(ZERO, mem->kin_pp);
   MFEM_ASSERT(flag >= 0, "KINInit() failed!");
   kinCopyInit(&backup, mem);

   // Delete the allocated data in y.
   nv::Destroy(y);

   // The 'user_data' in KINSOL will be the pointer 'this'.
   flag = KINSetUserData(sundials_mem, this);
   MFEM_ASSERT(flag >= 0, "KINSetUserData() failed!");

   if (!prec)
   {
      // Set scaled preconditioned GMRES linear solver.
      flag = KINSpgmr(sundials_mem, 0);
      MFEM_ASSERT(flag >= 0, "KINSpgmr() failed!");
      if (use_oper_grad)
      {
         // Define the Jacobian action.
         flag = KINSpilsSetJacTimesVecFn(sundials_mem, KinSolver::GradientMult);
         MFEM_ASSERT(flag >= 0, "KINSpilsSetJacTimesVecFn() failed!");
      }
   }
}

void KinSolver::SetSolver(Solver &solver)
{
   prec = &solver;

   KINMem mem = Mem(this);

   mem->kin_linit  = NULL;
   mem->kin_lsetup = KinSolver::LinSysSetup;
   mem->kin_lsolve = KinSolver::LinSysSolve;
   mem->kin_lfree  = NULL;
   mem->kin_lmem   = this;
   mem->kin_setupNonNull = TRUE;
   // Set mem->kin_inexact_ls? How?
}

void KinSolver::SetScaledStepTol(double sstol)
{
   Mem(this)->kin_scsteptol = sstol;
}

void KinSolver::SetMaxSetupCalls(int max_calls)
{
   Mem(this)->kin_msbset = max_calls;
}

void KinSolver::Mult(const Vector &b, Vector &x) const
{
  TMult(b, x);
}

#ifdef MFEM_USE_OCCA
void KinSolver::Mult(const OccaVector &b, OccaVector &x) const
{
  TMult(b, x);
}
#endif

void KinSolver::Mult(Vector &x,
                     const Vector &x_scale, const Vector &fx_scale) const
{
  TMult(x, x_scale, fx_scale);
}

#ifdef MFEM_USE_OCCA
void KinSolver::Mult(OccaVector &x,
                     const OccaVector &x_scale, const OccaVector &fx_scale) const
{
  TMult(x, x_scale, fx_scale);
}
#endif

KinSolver::~KinSolver()
{
   N_VDestroy(y);
   N_VDestroy(y_scale);
   N_VDestroy(f_scale);
   KINFree(&sundials_mem);
}

} // namespace mfem

#endif
