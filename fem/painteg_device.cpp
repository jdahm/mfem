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
#include <omp.h>

namespace mfem
{

template <int D, typename T = double> class TensorArray;

template <typename T>
class TensorArray<1,T>
{
   T *d;
   const int n1;
public:
   TensorArray(const T *_d, const int _n1) :
      d(const_cast<T*>(_d)), n1(_n1) { }

   TensorArray(const Vector &v) :
      d(v.GetData()), n1(v.Size()) { }

   operator T*() const { return d; }

   T& operator[](const int i) { return d[i]; }

   T& operator()(const int i1)
   { return d[i1]; }

   const T& operator()(const int i1) const
   { return d[i1]; }

   const TensorArray& operator=(const T& v)
   { for (int i = 0; i < n1; i++) d[i] = v; return *this; }
};

template <typename T>
class TensorArray<2,T>
{
   T *d;
   const int n1, n2;
public:
   TensorArray(const T *_d, const int _n1, const int _n2) :
      d(const_cast<T*>(_d)), n1(_n1), n2(_n2) { }

   TensorArray(DenseMatrix &m) :
      d(m.GetData()), n1(m.Height()), n2(m.Width()) { }

   operator T*() const { return d; }

   T& operator[](const int i) { return d[i]; }

   T& operator()(const int i1, const int i2)
   { return d[i2 * n1 + i1]; }

   const T& operator()(const int i1, const int i2) const
   { return d[i2 * n1 + i1]; }

   const TensorArray& operator=(const T& v)
   { for (int i = 0; i < n1*n2; i++) d[i] = v; return *this; }
};

template <typename T>
class TensorArray<3,T>
{
   T *d;
   const int n1, n2, n3;
public:
   TensorArray(const T *_d, const int _n1, const int _n2, const int _n3) :
      d(const_cast<T*>(_d)), n1(_n1), n2(_n2), n3(_n3) { }

   TensorArray(DenseTensor &t) :
      d(t.GetData(0)), n1(t.SizeI()), n2(t.SizeJ()), n3(t.SizeK()) { }

   operator T*() const { return d; }

   T& operator[](const int i) { return d[i]; }

   T& operator()(const int i1, const int i2, const int i3)
   { return d[(i3 * n2 + i2) * n1 + i1]; }

   const T& operator()(const int i1, const int i2, const int i3) const
   { return d[(i3 * n2 + i2) * n1 + i1]; }

   const TensorArray& operator=(const T& v)
   { for (int i = 0; i < n1*n2*n3; i++) d[i] = v; return *this; }
};

template <typename T>
class TensorArray<4,T>
{
   T *d;
   const int n1, n2, n3, n4;
public:
   TensorArray(const T *_d, const int _n1, const int _n2, const int _n3, const int _n4) :
      d(const_cast<T*>(_d)), n1(_n1), n2(_n2), n3(_n3), n4(_n4) { }

   operator T*() const { return d; }

   T& operator[](const int i) { return d[i]; }

   T& operator()(const int i1, const int i2, const int i3, const int i4)
   { return d[((i4 * n3 + i3) * n2 + i2) * n1 + i1]; }

   const T& operator()(const int i1, const int i2, const int i3, const int i4) const
   { return d[((i4 * n3 + i3) * n2 + i2) * n1 + i1]; }

   const TensorArray& operator=(const T& v)
   { for (int i = 0; i < n1*n2*n3*n4; i++) d[i] = v; return *this; }
};


void PADiffusionIntegrator::MultSeg_Device(const Vector &V, Vector &U)
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
   const int batchsize = 128;

   const int vdim = fes->GetVDim();
   MFEM_ASSERT(vdim == 1, "");

#pragma omp target teams                                \
   thread_limit(msize)                                  \
   is_device_ptr(data_d0, data_V, data_U, ds1d)
   {
      double s_dshape1d[50];
      double s_grad[50];

#pragma omp distribute
      for (int ebatch = 0; ebatch < NE/batchsize; ++ebatch)
      {

#pragma omp parallel num_threads(batchsize)
         {
#pragma omp for
            for (int x = 0; x < msize; ++x)
               for (int id = x; id < dofs1d * quads1d; id += msize)
               {
                  s_dshape1d[id] = ds1d[id];
               }
#pragma omp barrier

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

void PADiffusionIntegrator::MultQuad_Device(const Vector &V, Vector &U)
{
   const int dim = 2;
   const int terms = dim*(dim+1)/2;
   const int vdim = fes->GetVDim();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int msize = std::max(dofs1d, quads1d);

   const int dofs   = dofs1d * dofs1d;
   const int quads  = IntRule->GetNPoints();

   const double *data_d0 = Dtensor.GetData(0);

   const double *data_V = V.GetData();
   double *data_U = U.GetData();
   const double *s1d = shape1d.GetData();
   const double *ds1d = dshape1d.GetData();

   const int NE = fes->GetNE();

   MFEM_ASSERT(vdim == 1, "");

#pragma omp target teams                                \
   thread_limit(msize)                                  \
   is_device_ptr(data_d0, data_V, data_U, s1d, ds1d)
   {
      double s_shape1d[50];
      double s_dshape1d[50];
      double s_xy[50];
      double s_xDy[50];
      double s_grad[100];

#pragma omp distribute
      for (int e = 0; e < NE; ++e)
      {
#pragma omp parallel num_threads(msize)
         {
            const int d_offset = e * quads * terms;
            const double *data_d = data_d0 + d_offset;

            const int e_offset = dofs * e;
            const double *Ve = data_V + e_offset;
            double *Ue = data_U + e_offset;

            double r_x[11];
            double r_y[11];
#pragma omp for
            for (int x = 0; x < msize; ++x)
               for (int id = x; id < dofs1d * quads1d; id += msize)
               {
                  s_shape1d[id]  = s1d[id];
                  s_dshape1d[id] = ds1d[id];
               }
#pragma omp barrier

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
#pragma omp barrier

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
#pragma omp barrier

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
#pragma omp barrier

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

void PADiffusionIntegrator::MultHex_Device(const Vector &V, Vector &U)
{
   const int dim = 3;
   const int terms = dim*(dim+1)/2;
   const int vdim = fes->GetVDim();

   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int msize = std::max(dofs1d, quads1d);

   const int dofs   = dofs1d * dofs1d * dofs1d;
   const int quads  = IntRule->GetNPoints();

   const double *data_d0 = Dtensor.GetData(0);

   const double *data_V = V.GetData();
   double *data_U = U.GetData();
   const double *s1d = shape1d.GetData();
   const double *ds1d = dshape1d.GetData();

   const int NE = fes->GetNE();

   MFEM_ASSERT(vdim == 1, "");

#pragma omp target teams                        \
   thread_limit(msize*msize)                    \
   is_device_ptr(data_d0, data_V, data_U, s1d, ds1d)
   {
      double s_shape1d[50];
      double s_dshape1d[50];
      double s_z[50];
      double s_Dz[50];
      double s_xyDz[50];

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
            double r_qz[11];
            double r_qDz[11];
            double r_dDxyz[11];
            double r_dxDyz[11];
            double r_dxyDz[11];

            const int tid = omp_get_thread_num();
            const int tid_dx = (tid % dofs1d);
            const int tid_dy = (tid / dofs1d);

            const int tid_qx = (tid % quads1d);
            const int tid_qy = (tid / quads1d);

#pragma omp for collapse(2)
            for (int x = 0; x < msize; ++x)
               for (int id = x; id < dofs1d * quads1d; id += msize)
               {
                  s_shape1d[id]  = s1d[id];
                  s_dshape1d[id] = ds1d[id];
               }
#pragma omp barrier

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

            for (int dz = 0; dz < dofs1d; ++dz)
            {
               const double s = Ve[tid_dx + (tid_dy + dz * dofs1d) * dofs1d];
               for (int qz = 0; qz < quads1d; ++qz)
               {
                  r_qz[qz]  += s * s_shape1d[dz + qz * dofs1d];
                  r_qDz[qz] += s * s_dshape1d[dz + qz * dofs1d];
               }
            }

            // For each xy plane
            for (int qz = 0; qz < quads1d; ++qz)
            {
               s_z[tid_dx + tid_dy * dofs1d] = r_qz[qz];
               s_Dz[tid_dx + tid_dy * dofs1d] = r_qDz[qz];
#pragma omp barrier

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

            // Iterate over xy planes to compute solution
            for (int dz = 0; dz < dofs1d; ++dz)
            {
               s_z[tid_qx + tid_qy * quads1d] = r_dDxyz[dz];
               s_Dz[tid_qx + tid_qy * quads1d] = r_dxDyz[dz];
               s_xyDz[tid_qx + tid_qy * quads1d] = r_dxyDz[dz];
#pragma omp barrier

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

void PAMassIntegrator::MultSeg_Device(const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();

   const int dofs = dofs1d;
   const int quads = quads1d;
   const int vdim = fes->GetVDim();

   const double *data_d0 = Dtensor.GetData(0);

#pragma omp parallel
   {
      double *data_q = new double[quads1d];
      TensorArray<1> Q(data_q, quads1d);
#pragma omp for
      for (int e = 0; e < fes->GetNE(); ++e)
      {
         for (int vd = 0; vd < vdim; ++vd)
         {
            const int e_offset = dofs * (vdim * e + vd);
            const TensorArray<1> Vmat(V.GetData() + e_offset, dofs1d);
            TensorArray<1> Umat(U.GetData() + e_offset, dofs1d);

            Q = 0.;
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               const double v = Vmat(j1);
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  Q(k1) += v * shape1d(j1, k1);
               }
            }

            const int d_offset = e * quads;
            const double *data_d = data_d0 + d_offset;
            for (int k = 0; k < quads; ++k) { data_q[k] *= data_d[k]; }

            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               const double q = Q(k1);
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  Umat(i1) += q * shape1d(i1, k1);
               }
            }
         }
      }
      delete [] data_q;
   }
}

void PAMassIntegrator::MultQuad_Device(const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int msize = std::max(dofs1d, quads1d);

   const int dofs   = dofs1d * dofs1d;
   const int quads  = IntRule->GetNPoints();
   const int vdim = fes->GetVDim();

   const double *data_d0 = Dtensor.GetData(0);

#pragma omp parallel
   {
      double *data_q = new double[msize];
      double *data_qq = new double[quads1d * quads1d];
      TensorArray<1> Q(data_q, msize);
      TensorArray<2> QQ(data_qq, quads1d, quads1d);
#pragma omp for
      for (int e = 0; e < fes->GetNE(); ++e)
      {
         for (int vd = 0; vd < vdim; ++vd)
         {
            const int e_offset = dofs * (vdim * e + vd);
            const TensorArray<2> Vmat(V.GetData() + e_offset, dofs1d, dofs1d);
            TensorArray<2> Umat(U.GetData() + e_offset, dofs1d, dofs1d);

            QQ = 0.;
            for (int j2 = 0; j2 < dofs1d; ++j2)
            {
               Q = 0.;
               for (int j1 = 0; j1 < dofs1d; ++j1)
               {
                  const double v = Vmat(j1, j2);
                  for (int k1 = 0; k1 < quads1d; ++k1)
                  {
                     Q(k1) += v * shape1d(j1, k1);
                  }
               }
               for (int k2 = 0; k2 < quads1d; ++k2)
               {
                  const double s = shape1d(j2, k2);
                  for (int k1 = 0; k1 < quads1d; ++k1)
                  {
                     QQ(k1, k2) += Q(k1) * s;
                  }
               }
            }

            // QQ_c_k1_k2 = Dmat_c_d_k1_k2 * QQ_d_k1_k2
            // NOTE: (k1, k2) = k -- 1d index over tensor product of quad points
            const int d_offset = e * quads;
            const double *data_d = data_d0 + d_offset;
            for (int k = 0; k < quads; ++k) { data_qq[k] *= data_d[k]; }

            for (int k2 = 0; k2 < quads1d; ++k2)
            {
               Q = 0.;
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  const double q = QQ(k1, k2);
                  for (int i1 = 0; i1 < dofs1d; ++i1)
                  {
                     Q(i1) += q * shape1d(i1, k1);
                  }
               }
               for (int i2 = 0; i2 < dofs1d; ++i2)
               {
                  const double s = shape1d(i2, k2);
                  for (int i1 = 0; i1 < dofs1d; ++i1)
                  {
                     Umat(i1, i2) += Q(i1) * s;
                  }
               }
            }
         }
      }
      delete [] data_q;
      delete [] data_qq;
   }
}

void PAMassIntegrator::MultHex_Device(const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int msize = std::max(dofs1d, quads1d);

   const int dofs   = dofs1d * dofs1d * dofs1d;
   const int quads  = IntRule->GetNPoints();
   const int vdim = fes->GetVDim();

   const double *data_d0 = Dtensor.GetData(0);

#pragma omp parallel
   {
      double *data_q = new double[msize];
      double *data_qq = new double[msize * msize];
      double *data_qqq = new double[quads1d * quads1d * quads1d];
      TensorArray<1> Q(data_q, msize);
      TensorArray<2> QQ(data_qq, msize, msize);
      TensorArray<3> QQQ(data_qqq, quads1d, quads1d, quads1d);
#pragma omp for
      for (int e = 0; e < fes->GetNE(); ++e)
      {
         for (int vd = 0; vd < vdim; ++vd)
         {
            const int e_offset = dofs * (vdim * e + vd);
            const TensorArray<3> Vmat(V.GetData() + e_offset, dofs1d, dofs1d, dofs1d);
            TensorArray<3> Umat(U.GetData() + e_offset, dofs1d, dofs1d, dofs1d);

            // QQQ_k1_k2_k3 = shape_j1_k1 * shape_j2_k2  * shape_j3_k3  * Vmat_j1_j2_j3
            QQQ = 0.;
            for (int j3 = 0; j3 < dofs1d; ++j3)
            {
               QQ = 0.;
               for (int j2 = 0; j2 < dofs1d; ++j2)
               {
                  Q = 0.;
                  for (int j1 = 0; j1 < dofs1d; ++j1)
                  {
                     const double v = Vmat(j1, j2, j3);
                     for (int k1 = 0; k1 < quads1d; ++k1)
                     {
                        Q(k1) += v * shape1d(j1, k1);
                     }
                  }
                  for (int k2 = 0; k2 < quads1d; ++k2)
                  {
                     const double s = shape1d(j2, k2);
                     for (int k1 = 0; k1 < quads1d; ++k1)
                     {
                        QQ(k1, k2) += Q(k1) * s;
                     }
                  }
               }
               for (int k3 = 0; k3 < quads1d; ++k3)
               {
                  const double s = shape1d(j3, k3);
                  for (int k2 = 0; k2 < quads1d; ++k2)
                     for (int k1 = 0; k1 < quads1d; ++k1)
                     {
                        QQQ(k1, k2, k3) += QQ(k1, k2) * s;
                     }
               }
            }

            // QQQ_k1_k2_k3 = Dmat_k1_k2_k3 * QQQ_k1_k2_k3
            // NOTE: (k1, k2, k3) = q -- 1d quad point index
            const int d_offset = e * quads;
            const double *data_d = data_d0 + d_offset;
            for (int k = 0; k < quads; ++k) { data_qqq[k] *= data_d[k]; }

            // Apply transpose of the first operator that takes V -> QQQ -- QQQ -> U
            for (int k3 = 0; k3 < quads1d; ++k3)
            {
               QQ = 0.;
               for (int k2 = 0; k2 < quads1d; ++k2)
               {
                  Q = 0.;
                  for (int k1 = 0; k1 < quads1d; ++k1)
                  {
                     const double q = QQQ(k1, k2, k3);
                     for (int i1 = 0; i1 < dofs1d; ++i1)
                     {
                        Q(i1) += q * shape1d(i1, k1);
                     }
                  }
                  for (int i2 = 0; i2 < dofs1d; ++i2)
                  {
                     const double s = shape1d(i2, k2);
                     for (int i1 = 0; i1 < dofs1d; ++i1)
                     {
                        QQ(i1, i2) += Q(i1) * s;
                     }
                  }
               }
               for (int i3 = 0; i3 < dofs1d; ++i3)
               {
                  const double s = shape1d(i3, k3);
                  for (int i2 = 0; i2 < dofs1d; ++i2)
                     for (int i1 = 0; i1 < dofs1d; ++i1)
                     {
                        Umat(i1, i2, i3) += s * QQ(i1, i2);
                     }
               }
            }
         }
      }
      delete [] data_q;
      delete [] data_qq;
      delete [] data_qqq;
   }
}

}
