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

// Implementation of FESpaceIntegrators.

#include "fem.hpp"
#include <stdio.h>

// Remove this when the 3D integrators are "fixed"
#if defined(MFEM_USE_OPENMP)
#include <omp.h>

static const int max_order = 6;
static const int max_quads1d = max_order+2;
static const int max_dofs1d = max_order+1;
static const int max_dofs1d_dofs1d = max_quads1d * max_dofs1d;
static const int max_dofs1d_quads1d = max_quads1d * max_dofs1d;
static const int max_quads1d_quads1d = max_quads1d * max_dofs1d;

static const int max_batchsize = 128;

namespace mfem
{

void FESDiffusionIntegrator::MultSeg_Target(const Vector &V, Vector &U)
{
   const int dim = 1;
   const int terms = dim*(dim+1)/2;
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int msize = std::max(dofs1d, quads1d);

   const int dofs = dofs1d;
   const int quads = quads1d;

   const double *data_d0 = Dtensor.GetData(0);

   const double *data_V = V.GetData();
   double *data_U = U.GetData();
   const double *ds1d = dshape1d.GetData();

   const int NE = fes->GetNE();
   const int batchsize = std::min(max_batchsize, (int) std::ceil((double) NE / 20));
   const int num_batches = std::ceil((double) NE / batchsize);

   const int vdim = fes->GetVDim();
   MFEM_ASSERT(vdim == 1, "");

#pragma omp target teams                                \
   thread_limit(batchsize)                              \
   is_device_ptr(data_d0, data_V, data_U, ds1d)
   {
      double s_dshape1d[max_dofs1d_quads1d];
      double s_grad[max_quads1d];

#pragma omp distribute
      for (int ebatch = 0; ebatch < num_batches; ++ebatch)
      {
#pragma omp parallel
         {

#pragma omp for
            for (int x = 0; x < msize; ++x)
               for (int id = x; id < dofs1d * quads1d; id += msize)
               {
                  s_dshape1d[id] = ds1d[id];
               }

#pragma omp for
            for (int el = 0; el < batchsize; ++el)
            {
               const int e = ebatch * batchsize + el;
               if (e < NE)
               {
                  const double *Ve = data_V + e * dofs;
                  double *Ue = data_U + e * dofs;

                  for (int qx = 0; qx < quads1d; ++qx) s_grad[qx] = 0;
                  for (int dx = 0; dx < dofs1d; ++dx)
                  {
                     const double s = Ve[dx];
                     for (int qx = 0; qx < quads1d; ++qx)
                        s_grad[qx] += s * s_dshape1d[dx + qx * dofs1d];
                  }

                  const int d_offset = e * quads * terms;
                  const double *data_d = data_d0 + d_offset;
                  for (int qx = 0; qx < quads1d; ++qx)
                     s_grad[qx] *= data_d[terms * qx + 0];

                  for (int dx = 0; dx < dofs1d; ++dx)
                  {
                     double s = 0;
                     for (int qx = 0; qx < quads1d; ++qx)
                        s += s_grad[qx] * s_dshape1d[dx + qx * dofs1d];
                     Ue[dx] += s;
                  }
               }
            }
         }
      }
   }
}

void FESDiffusionIntegrator::MultQuad_Target(const Vector &V, Vector &U)
{
   const int dim = 2;
   const int terms = dim*(dim+1)/2;
   const int vdim = fes->GetVDim();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int msize = std::max(dofs1d, quads1d);

   const int dofs   = dofs1d * dofs1d;
   const int quads  = quads1d * quads1d;

   const double *data_d0 = Dtensor.GetData(0);

   const double *data_V = V.GetData();
   double *data_U = U.GetData();
   const double *s1d = shape1d.GetData();
   const double *ds1d = dshape1d.GetData();

   const int NE = fes->GetNE();
   const int batchsize = std::min(max_batchsize, (int) std::ceil((double) NE / 20));
   const int num_batches = std::ceil((double) NE / batchsize);

   MFEM_ASSERT(vdim == 1, "");

#pragma omp target teams                                \
   thread_limit(msize)                                  \
   is_device_ptr(data_d0, data_V, data_U, s1d, ds1d)
   {
      double s_shape1d[max_dofs1d_quads1d];
      double s_dshape1d[max_dofs1d_quads1d];
      double s_xy[max_dofs1d_quads1d];
      double s_xDy[max_dofs1d_quads1d];
      double s_grad[2 * max_quads1d_quads1d];

#pragma omp distribute
      for (int ebatch = 0; ebatch < num_batches; ++ebatch)
      {
#pragma omp parallel
         {
            double r_x[max_dofs1d];
            double r_y[max_dofs1d];

#pragma omp for
            for (int x = 0; x < msize; ++x)
               for (int id = x; id < dofs1d * quads1d; id += msize)
               {
                  s_shape1d[id]  = s1d[id];
                  s_dshape1d[id] = ds1d[id];
               }

            for (int el = 0; el < batchsize; ++el)
            {
               const int e = ebatch * batchsize + el;
               if (e < NE)
               {
                  const int d_offset = e * quads * terms;
                  const double *data_d = data_d0 + d_offset;

                  const int e_offset = dofs * e;
                  const double *Ve = data_V + e_offset;
                  double *Ue = data_U + e_offset;

#pragma omp for
                  for (int dx = 0; dx < dofs1d; ++dx)
                  {
                     for (int dy = 0; dy < dofs1d; ++dy) r_x[dy] = Ve[dx + dy * dofs1d];
                     for (int qy = 0; qy < quads1d; ++qy)
                     {
                        double xy = 0;
                        double xDy = 0;
                        for (int dy = 0; dy < dofs1d; ++dy)
                        {
                           xy  += r_x[dy] * s_shape1d[dy + qy * dofs1d];
                           xDy += r_x[dy] * s_dshape1d[dy + qy * dofs1d];
                        }
                        s_xy[dx + qy * dofs1d]  = xy;
                        s_xDy[dx + qy * dofs1d] = xDy;
                     }
                  }

#pragma omp for collapse(2)
                  for (int qy = 0; qy < quads1d; ++qy)
                     for (int qx = 0; qx < quads1d; ++qx)
                     {
                        double gradX = 0, gradY = 0;
                        for (int dx = 0; dx < dofs1d; ++dx)
                        {
                           gradX += s_xy[dx + qy * dofs1d]  * s_dshape1d[dx + qx * dofs1d];
                           gradY += s_xDy[dx + qy * dofs1d] * s_shape1d[dx + qx * dofs1d];
                        }

                        const int q = qy * quads1d + qx;
                        const double O11 = data_d[terms*q + 0];
                        const double O12 = data_d[terms*q + 1];
                        const double O22 = data_d[terms*q + 2];

                        s_grad[0 * quads + qx + qy * quads1d] = (O11 * gradX) + (O12 * gradY);
                        s_grad[1 * quads + qx + qy * quads1d] = (O12 * gradX) + (O22 * gradY);
                     }

#pragma omp for
                  for (int qx = 0; qx < quads1d; ++qx)
                  {
                     for (int qy = 0; qy < quads1d; ++qy)
                     {
                        r_x[qy] = s_grad[0 * quads + qx + qy * quads1d];
                        r_y[qy] = s_grad[1 * quads + qx + qy * quads1d];
                     }
                     for (int dy = 0; dy < dofs1d; ++dy)
                     {
                        double xy  = 0;
                        double xDy = 0;
                        for (int qy = 0; qy < quads1d; ++qy)
                        {
                           xy  += r_x[qy] * s_shape1d[dy + qy * dofs1d];
                           xDy += r_y[qy] * s_dshape1d[dy + qy * dofs1d];
                        }
                        s_xy[dy + qx * dofs1d] = xy;
                        s_xDy[dy + qx * dofs1d] = xDy;
                     }
                  }

#pragma omp for collapse(2)
                  for (int dx = 0; dx < dofs1d; ++dx)
                     for (int dy = 0; dy < dofs1d; ++dy)
                     {
                        double s = 0;
                        for (int qx = 0; qx < quads1d; ++qx)
                           s += ((s_xy[dy + qx * dofs1d] * s_dshape1d[dx + qx * dofs1d]) +
                                 (s_xDy[dy + qx * dofs1d] * s_shape1d[dx + qx * dofs1d]));
                        Ue[dx + dy * dofs1d] += s;
                     }
               }
            }
         }
      }
   }
}

void FESDiffusionIntegrator::MultHex_Target(const Vector &V, Vector &U)
{
   const int dim = 3;
   const int terms = dim*(dim+1)/2;
   const int vdim = fes->GetVDim();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int msize = std::max(dofs1d, quads1d);

   const int dofs   = dofs1d * dofs1d * dofs1d;
   const int quads  = quads1d * quads1d * quads1d;

   const double *data_d0 = Dtensor.GetData(0);

   const double *data_V = V.GetData();
   double *data_U = U.GetData();
   const double *s1d = shape1d.GetData();
   const double *ds1d = dshape1d.GetData();

   const int NE = fes->GetNE();

   MFEM_ASSERT(vdim == 1, "");

#pragma omp target teams                        \
   is_device_ptr(data_d0, data_V, data_U, s1d, ds1d)
   {
      double s_shape1d[max_dofs1d_quads1d];
      double s_dshape1d[max_dofs1d_quads1d];
      double s_z[max_dofs1d_dofs1d];
      double s_Dz[max_dofs1d_dofs1d];
      double s_xyDz[max_dofs1d_quads1d];

#pragma omp distribute
      for (int e = 0; e < NE; ++e)
      {
         const int d_offset = e * quads * terms;
         const double *data_d = data_d0 + d_offset;

         const int e_offset = dofs * e;
         const double *Ve = data_V + e_offset;
         double *Ue = data_U + e_offset;

#pragma omp parallel num_threads(msize*msize)
         {
            // Thread-private storage
            double r_qz[max_quads1d];
            double r_qDz[max_quads1d];
            double r_dDxyz[max_dofs1d];
            double r_dxDyz[max_dofs1d];
            double r_dxyDz[max_dofs1d];

            const int tid = omp_get_thread_num();
            const int tid_dx = (tid % dofs1d);
            const int tid_dy = (tid / dofs1d);

            const int tid_qx = (tid % quads1d);
            const int tid_qy = (tid / quads1d);

#pragma omp for
            for (int id = 0; id < quads1d * dofs1d; ++id)
               if (id < quads1d * dofs1d)
               {
                  s_shape1d[id]  = s1d[id];
                  s_dshape1d[id] = ds1d[id];
               }

            for (int qz = 0; qz < quads1d; ++qz)
            {
               r_qz[qz] = 0;
               r_qDz[qz] = 0;
            }
            for (int dz = 0; dz < dofs1d; ++dz) {
               r_dDxyz[dz] = 0;
               r_dxDyz[dz] = 0;
               r_dxyDz[dz] = 0;
            }
            if (tid < dofs1d * dofs1d)
            {
               for (int dz = 0; dz < dofs1d; ++dz)
               {
                  const double s = Ve[tid_dx + (tid_dy + dz * dofs1d) * dofs1d];
                  for (int qz = 0; qz < quads1d; ++qz)
                  {
                     r_qz[qz]  += s * s_shape1d[dz + qz * dofs1d];
                     r_qDz[qz] += s * s_dshape1d[dz + qz * dofs1d];
                  }
               }
            }

            // For each xy plane
            for (int qz = 0; qz < quads1d; ++qz)
            {
               if (tid < dofs1d * dofs1d)
               {
                  s_z[tid_dx + tid_dy * dofs1d] = r_qz[qz];
                  s_Dz[tid_dx + tid_dy * dofs1d] = r_qDz[qz];
               }
#pragma omp barrier

               if (tid < quads1d * quads1d)
               {
                  double Dxyz = 0;
                  double xDyz = 0;
                  double xyDz = 0;
                  for (int dy = 0; dy < dofs1d; ++dy)
                  {
                     const double wy  = s_shape1d[dy + tid_qy * dofs1d];
                     const double wDy = s_dshape1d[dy + tid_qy * dofs1d];
                     for (int dx = 0; dx < dofs1d; ++dx)
                     {
                        const double wx  = s_shape1d[dx + tid_qx * dofs1d];
                        const double wDx = s_dshape1d[dx + tid_qx * dofs1d];
                        const double z  = s_z[dx + dy * dofs1d];
                        const double Dz = s_Dz[dx + dy * dofs1d];
                        Dxyz += wDx * wy  * z;
                        xDyz += wx  * wDy * z;
                        xyDz += wx  * wy  * Dz;
                     }
                  }

                  const int q = tid_qx + (tid_qy + qz * quads1d) * quads1d;
                  const double O11 = data_d[terms*q + 0];
                  const double O12 = data_d[terms*q + 1];
                  const double O13 = data_d[terms*q + 2];
                  const double O22 = data_d[terms*q + 3];
                  const double O23 = data_d[terms*q + 4];
                  const double O33 = data_d[terms*q + 5];

                  const double qDxyz = (O11 * Dxyz) + (O12 * xDyz) + (O13 * xyDz);
                  const double qxDyz = (O12 * Dxyz) + (O22 * xDyz) + (O23 * xyDz);
                  const double qxyDz = (O13 * Dxyz) + (O23 * xDyz) + (O33 * xyDz);

                  for (int dz = 0; dz < dofs1d; ++dz) {
                     const double wz  = s_shape1d[dz + qz * dofs1d];
                     const double wDz = s_dshape1d[dz + qz * dofs1d];
                     r_dDxyz[dz] += wz  * qDxyz;
                     r_dxDyz[dz] += wz  * qxDyz;
                     r_dxyDz[dz] += wDz * qxyDz;
                  }
               }
            }

            // Iterate over xy planes to compute solution
            for (int dz = 0; dz < dofs1d; ++dz)
            {
               if (tid < quads1d * quads1d)
               {
                  s_z[tid_qx + tid_qy * quads1d] = r_dDxyz[dz];
                  s_Dz[tid_qx + tid_qy * quads1d] = r_dxDyz[dz];
                  s_xyDz[tid_qx + tid_qy * quads1d] = r_dxyDz[dz];
               }
#pragma omp barrier

               if (tid < dofs1d * dofs1d)
               {
                  // Finalize solution in xy plane
                  double solZ = 0;
                  for (int qy = 0; qy < quads1d; ++qy) {
                     const double wy  = s_shape1d[tid_dy + qy * dofs1d];
                     const double wDy = s_dshape1d[tid_dy + qy * dofs1d];
                     for (int qx = 0; qx < quads1d; ++qx) {
                        const double wx  = s_shape1d[tid_dx + qx * dofs1d];
                        const double wDx = s_dshape1d[tid_dx + qx * dofs1d];
                        const double Dxyz = s_z[qx + qy * quads1d];
                        const double xDyz = s_Dz[qx + qy * quads1d];
                        const double xyDz = s_xyDz[qx + qy * quads1d];
                        solZ += ((wDx * wy  * Dxyz) +
                                 (wx  * wDy * xDyz) +
                                 (wx  * wy  * xyDz));
                     }
                  }
                  Ue[tid_dx + (tid_dy + dz * dofs1d) * dofs1d] += solZ;
               }
            }
         }
      }
   }
}

void FESMassIntegrator::MultSeg_Target(const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int msize = std::max(dofs1d, quads1d);

   const int dofs = dofs1d;
   const int quads = quads1d;
   const int vdim = fes->GetVDim();
   MFEM_ASSERT(vdim == 1, "");

   const double *data_d0 = Dtensor.GetData(0);

   const double *data_V = V.GetData();
   double *data_U = U.GetData();
   const double *s1d = shape1d.GetData();

   const int NE = fes->GetNE();
   const int batchsize = std::min(max_batchsize, (int) std::ceil((double) NE / 20));
   const int num_batches = std::ceil((double) NE / batchsize);

#pragma omp target teams                                \
   thread_limit(batchsize)                              \
   is_device_ptr(data_d0, data_V, data_U, s1d)
   {
      double s_shape1d[max_dofs1d_quads1d];
      double s_data[max_quads1d];

#pragma omp distribute
      for (int ebatch = 0; ebatch < num_batches; ++ebatch)
      {
#pragma omp parallel
         {

#pragma omp for
            for (int x = 0; x < msize; ++x)
               for (int id = x; id < dofs1d * quads1d; id += msize)
               {
                  s_shape1d[id] = s1d[id];
               }

#pragma omp for
            for (int el = 0; el < batchsize; ++el)
            {
               const int e = ebatch * batchsize + el;
               if (e < NE)
               {
                  const double *data_d = data_d0 + e * quads;
                  const double *Ve = data_V + e * dofs;
                  double *Ue = data_U + e * dofs;
                  for (int dx = 0; dx < dofs1d; ++dx)
                  {
                     const double s = Ve[dx];
                     for (int qx = 0; qx < quads1d; ++qx)
                        s_data[qx] += s * s_shape1d[dx + qx * dofs1d];
                  }

                  for (int qx = 0; qx < quads1d; ++qx)
                     s_data[qx] *= data_d[qx];

                  for (int dx = 0; dx < dofs1d; ++dx)
                  {
                     double s = 0;
                     for (int qx = 0; qx < quads1d; ++qx)
                        s += s_data[qx] * s_shape1d[dx + qx * dofs1d];
                     Ue[dx] += s;
                  }
               }
            }
         }
      }
   }
}

void FESMassIntegrator::MultQuad_Target(const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int msize = std::max(dofs1d, quads1d);

   const int dofs   = dofs1d * dofs1d;
   const int quads  = quads1d * quads1d;
   const int vdim = fes->GetVDim();

   const double *data_d0 = Dtensor.GetData(0);

   const double *data_V = V.GetData();
   double *data_U = U.GetData();
   const double *s1d = shape1d.GetData();

   const int NE = fes->GetNE();
   const int batchsize = std::min(max_batchsize, (int) std::ceil((double) NE / 20));
   const int num_batches = std::ceil((double) NE / batchsize);

   MFEM_ASSERT(vdim == 1, "");

#pragma omp target teams                        \
   thread_limit(msize)                          \
   is_device_ptr(data_d0, data_V, data_U, s1d)
   {
      double s_shape1d[max_dofs1d_quads1d];
      double s_xy[max_dofs1d_quads1d];
      double s_xy2[max_quads1d_quads1d];

#pragma omp distribute
      for (int ebatch = 0; ebatch < num_batches; ++ebatch)
      {
#pragma omp parallel
         {
            double r_x[max_quads1d];

#pragma omp for
            for (int x = 0; x < msize; ++x)
               for (int id = x; id < dofs1d * quads1d; id += msize)
               {
                  s_shape1d[id] = s1d[id];
               }

            for (int el = 0; el < batchsize; ++el)
            {
               const int e = ebatch * batchsize + el;
               if (e < NE)
               {
                  const double *data_d = data_d0 + e * quads;
                  const double *Ve = data_V + e * dofs;
                  double *Ue = data_U + e * dofs;

#pragma omp for
                  for (int dx = 0; dx < dofs1d; ++dx)
                  {
                     for (int qy = 0; qy < quads1d; ++qy)
                        s_xy[dx + qy * dofs1d] = 0;

                     for (int dy = 0; dy < dofs1d; ++dy)
                        r_x[dy] = Ve[dx + dy * dofs1d];

                     for (int qy = 0; qy < quads1d; ++qy)
                     {
                        double xy = 0;
                        for (int dy = 0; dy < dofs1d; ++dy)
                           xy += r_x[dy] * s_shape1d[dy + qy *  dofs1d];
                        s_xy[dx + qy + dofs1d] = xy;
                     }
                  }

#pragma omp for
                  for (int qy = 0; qy < quads1d; ++qy)
                     for (int qx = 0; qx < quads1d; ++qx)
                     {
                        double s = 0;
                        for (int dx = 0; dx < dofs1d; ++dx)
                           s += s_xy[dx + qy * dofs1d] * s_shape1d[dx + qx * dofs1d];
                        s_xy2[qx + qy * quads1d] = s * data_d[qx + qy * quads1d];
                     }

#pragma omp for
                  for (int qx = 0; qx < quads1d; ++qx)
                  {
                     for (int dy = 0; dy < dofs1d; ++dy)
                        s_xy[dy + qx * dofs1d] = 0;

                     for (int qy = 0; qy < quads1d; ++qy)
                        r_x[qy] = s_xy2[qx + qy * quads1d];

                     for (int dy = 0; dy < dofs1d; ++dy)
                     {
                        double s = 0;
                        for (int qy = 0; qy < quads1d; ++qy)
                           s += r_x[qy] * s_shape1d[dy + qy * dofs1d];
                        s_xy[dy + qx * dofs1d] = s;
                     }
                  }

#pragma omp for
                  for (int dx = 0; dx < dofs1d; ++dx)
                     for (int dy = 0; dy < dofs1d; ++dy) {
                        double s = 0;
                        for (int qx = 0; qx < quads1d; ++qx)
                           s += s_xy[dy + qx * dofs1d] * s_shape1d[dx + qx * dofs1d];
                        Ue[dx + dy * dofs1d] += s;
                     }
               }
            }
         }
      }
   }
}

void FESMassIntegrator::MultHex_Target(const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int msize = std::max(dofs1d, quads1d);

   const int dofs   = dofs1d * dofs1d * dofs1d;
   const int quads  = quads1d * quads1d * quads1d;
   const int vdim = fes->GetVDim();

   const double *data_d0 = Dtensor.GetData(0);

   const double *data_V = V.GetData();
   double *data_U = U.GetData();
   const double *s1d = shape1d.GetData();

   const int NE = fes->GetNE();

   MFEM_ASSERT(vdim == 1, "");

#pragma omp target teams                        \
   is_device_ptr(data_d0, data_V, data_U, s1d)
   {
      double s_shape1d[max_dofs1d_quads1d];
      double s_xy[max_quads1d_quads1d];

#pragma omp distribute
      for (int e = 0; e < NE; e++)
      {
         const double *data_d = data_d0 + e * quads;
         const double *Ve = data_V + e * dofs;
         double *Ue = data_U + e * dofs;

#pragma omp parallel num_threads(msize*msize)
         {
            double r_z[max_quads1d];
            double r_z2[max_quads1d];

            const int tid = omp_get_thread_num();
            const int tid_dx = (tid % dofs1d);
            const int tid_dy = (tid / dofs1d);

            const int tid_qx = (tid % quads1d);
            const int tid_qy = (tid / quads1d);

#pragma omp for
            for (int x = 0; x < msize; ++x)
               for (int id = x; id < dofs1d * quads1d; id += msize)
               {
                  s_shape1d[id] = s1d[id];
               }

            for (int qz = 0; qz < quads1d; ++qz) r_z[qz] = r_z2[qz] = 0;

            if (tid < dofs1d * dofs1d)
            {

               for (int dz = 0; dz < dofs1d; ++dz)
               {
                  const double s = Ve[tid_dx + dofs1d * (tid_dy + dofs1d * dz)];
                  // Calculate D -> Q in the Z axis
                  for (int qz = 0; qz < quads1d; ++qz)
                     r_z[qz] += s * s_shape1d[dz + qz * dofs1d];
               }
            }

            // For each xy plane
            for (int qz = 0; qz < quads1d; ++qz) {
               // Fill xy plane at given z position
               if (tid < dofs1d * dofs1d) s_xy[tid_dx + tid_dy * dofs1d] = r_z[qz];
#pragma omp barrier

               // Calculate Dxyz, xDyz, xyDz in plane
               if (tid < quads1d * quads1d)
               {
                  double s = 0;
                  for (int dy = 0; dy < dofs1d; ++dy)
                  {
                     const double wy = s_shape1d[dy + tid_qy * dofs1d];
                     for (int dx = 0; dx < dofs1d; ++dx)
                     {
                        const double wx = s_shape1d[dx + tid_qx * dofs1d];
                        s += wx * wy * s_xy[dx + dy * dofs1d];
                     }
                  }

                  s *= data_d[tid_qx + quads1d * (tid_qy + quads1d * qz)];

                  for (int dz = 0; dz < dofs1d; ++dz)
                  {
                     const double wz  = s_shape1d[dz + qz * dofs1d];
                     r_z2[dz] += wz * s;
                  }
               }
            }
            // Iterate over xy planes to compute solution
            for (int dz = 0; dz < dofs1d; ++dz) {
               // Place xy plane in shared memory
               if (tid < quads1d * quads1d) s_xy[tid_qx + tid_qy * quads1d] = r_z2[dz];
#pragma omp barrier

               // Finalize solution in xy plane
               if (tid < dofs1d * dofs1d)
               {
                  double solZ = 0;
                  for (int qy = 0; qy < quads1d; ++qy)
                  {
                     const double wy = s_shape1d[tid_dy + qy * dofs1d];
                     for (int qx = 0; qx < quads1d; ++qx)
                     {
                        const double wx = s_shape1d[tid_dx + qx * dofs1d];
                        solZ += wx * wy * s_xy[qx + qy * quads1d];
                     }
                  }
                  Ue[tid_dx + dofs1d * (tid_dy + dofs1d * dz)] += solZ;
               }
            }
         }
      }
   }
}

}

#else

namespace mfem
{

void FESDiffusionIntegrator::MultSeg_Target(const Vector &V, Vector &U)
{
   mfem_error("Not supported");
}

void FESDiffusionIntegrator::MultQuad_Target(const Vector &V, Vector &U)
{
   mfem_error("Not supported");
}

void FESDiffusionIntegrator::MultHex_Target(const Vector &V, Vector &U)
{
   mfem_error("Not supported");
}

void FESMassIntegrator::MultSeg_Target(const Vector &V, Vector &U)
{
   mfem_error("Not supported");
}

void FESMassIntegrator::MultQuad_Target(const Vector &V, Vector &U)
{
   mfem_error("Not supported");
}

void FESMassIntegrator::MultHex_Target(const Vector &V, Vector &U)
{
   mfem_error("Not supported");
}

}

#endif
