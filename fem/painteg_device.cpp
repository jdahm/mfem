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

   const int dofs = dofs1d;
   const int quads = quads1d;
   const int vdim = fes->GetVDim();

   const double *data_d0 = Dtensor.GetData(0);

#pragma omp parallel
   {
      double *data_q = new double[quads1d * dim];
      TensorArray<2> Q(data_q, quads1d, dim);
#pragma omp for
      {
         for (int e = 0; e < fes->GetNE(); ++e)
         {
            for (int vd = 0; vd < vdim; ++vd)
            {
               const int e_offset = dofs * (vdim * e + vd);
               const TensorArray<1> Vmat(V.GetData() + e_offset, dofs1d);
               TensorArray<1> Umat(U.GetData() + e_offset, dofs1d);

               // Q_k1 = dshape_j1_k1 * Vmat_j1
               Q = 0.;
               for (int j1 = 0; j1 < dofs1d; ++j1)
               {
                  const double v = Vmat(j1);
                  for (int k1 = 0; k1 < quads1d; ++k1)
                  {
                     Q(k1, 0) += v * dshape1d(j1, k1);
                  }
               }

               const int d_offset = e * quads * terms;
               const double *data_d = data_d0 + d_offset;
               for (int k = 0; k < quads; ++k)
               {
                  data_q[k] *= data_d[k];
               }

               // Umat_k1 = dshape_j1_k1 * Q_k1
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  const double q = Q(k1, 0);
                  for (int i1 = 0; i1 < dofs1d; ++i1)
                  {
                     Umat(i1) += q * dshape1d(i1, k1);
                  }
               }
            }
         }
      }
      delete [] data_q;
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

#pragma omp target teams thread_limit(msize) is_device_ptr(data_d0, data_V, data_U, s1d, ds1d)
   {
      double s_shape1d[50];
      double s_dshape1d[50];
      double s_xy[50];
      double s_xDy[50];
      double s_grad[50];

#pragma omp distribute
      for (int e = 0; e < NE; ++e)
      {
#pragma omp parallel for num_threads(msize)
         for (int x = 0; x < msize; ++x)
            for (int id = x; id < dofs1d * quads1d; id += msize)
            {
               s_shape1d[id]  = s1d[id];
               s_dshape1d[id] = ds1d[id];
            }

         const int e_offset = dofs * e;
         const TensorArray<2> Vmat(data_V + e_offset, dofs1d, dofs1d);
         TensorArray<2> Umat(data_U + e_offset, dofs1d, dofs1d);
         const int d_offset = e * quads * terms;
         const double *data_d = data_d0 + d_offset;

#pragma omp parallel for num_threads(dofs1d)
         for (int dx = 0; dx < dofs1d; ++dx)
         {
            double r_x[11];
            for (int dy = 0; dy < dofs1d; ++dy) r_x[dy] = Vmat(dx, dy);
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

#pragma omp parallel for num_threads(quads1d)
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

#pragma omp parallel for num_threads(quads1d)
         for (int qx = 0; qx < quads1d; ++qx)
         {
            double r_x[11];
            double r_y[11];
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

#pragma omp parallel for num_threads(dofs1d)
         for (int dx = 0; dx < dofs1d; ++dx)
            for (int dy = 0; dy < dofs1d; ++dy)
            {
               double s = 0;
               for (int qx = 0; qx < quads1d; ++qx)
                  s += ((s_xy[dy + qx * dofs1d] * s_dshape1d[dx + qx * dofs1d]) +
                        (s_xDy[dy + qx * dofs1d] * s_shape1d[dx + qx * dofs1d]));
               Umat[dx + dy * dofs1d] += s;
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

#pragma omp parallel
   {
      double *data_q = new double[msize * dim];
      double *data_qq = new double[msize * msize * dim];
      double *data_qqq = new double[quads1d * quads1d * quads1d * dim];
      TensorArray<2> Q(data_q, msize, dim);
      TensorArray<3> QQ(data_qq, msize, msize, dim);
      TensorArray<3> QQQ0(data_qqq + 0*quads, quads1d, quads1d, quads1d);
      TensorArray<3> QQQ1(data_qqq + 1*quads, quads1d, quads1d, quads1d);
      TensorArray<3> QQQ2(data_qqq + 2*quads, quads1d, quads1d, quads1d);
#pragma omp for
      {
         for (int e = 0; e < fes->GetNE(); ++e)
         {
            for (int vd = 0; vd < vdim; ++vd)
            {
               const int e_offset = dofs * (vdim * e + vd);
               const TensorArray<3> Vmat(V.GetData() + e_offset, dofs1d, dofs1d, dofs1d);
               TensorArray<3> Umat(U.GetData() + e_offset, dofs1d, dofs1d, dofs1d);

               // QQQ_0_k1_k2_k3 = dshape_j1_k1 * shape_j2_k2  * shape_j3_k3  * Vmat_j1_j2_j3
               // QQQ_1_k1_k2_k3 = shape_j1_k1  * dshape_j2_k2 * shape_j3_k3  * Vmat_j1_j2_j3
               // QQQ_2_k1_k2_k3 = shape_j1_k1  * shape_j2_k2  * dshape_j3_k3 * Vmat_j1_j2_j3
               QQQ0 = 0.; QQQ1 = 0.; QQQ2 = 0.;
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
                           Q(k1, 0) += v * dshape1d(j1, k1);
                           Q(k1, 1) += v * shape1d(j1, k1);
                        }
                     }
                     for (int k2 = 0; k2 < quads1d; ++k2)
                     {
                        const double s = shape1d(j2, k2);
                        const double d = dshape1d(j2, k2);
                        for (int k1 = 0; k1 < quads1d; ++k1)
                        {
                           QQ(k1, k2, 0) += Q(k1, 0) * s;
                           QQ(k1, k2, 1) += Q(k1, 1) * d;
                           QQ(k1, k2, 2) += Q(k1, 1) * s;
                        }
                     }
                  }
                  for (int k3 = 0; k3 < quads1d; ++k3)
                  {
                     const double s = shape1d(j3, k3);
                     const double d = dshape1d(j3, k3);
                     for (int k2 = 0; k2 < quads1d; ++k2)
                        for (int k1 = 0; k1 < quads1d; ++k1)
                        {
                           QQQ0(k1, k2, k3) += QQ(k1, k2, 0) * s;
                           QQQ1(k1, k2, k3) += QQ(k1, k2, 1) * s;
                           QQQ2(k1, k2, k3) += QQ(k1, k2, 2) * d;
                        }
                  }
               }

               // QQQ_c_k1_k2_k3 = Dmat_c_d_k1_k2_k3 * QQQ_d_k1_k2_k3
               // NOTE: (k1, k2, k3) = q -- 1d quad point index
               const int d_offset = e * quads * terms;
               const double *data_d = data_d0 + d_offset;
               for (int k = 0; k < quads; ++k)
               {
                  const double D00 = data_d[terms*k + 0];
                  const double D01 = data_d[terms*k + 1];
                  const double D02 = data_d[terms*k + 2];
                  const double D11 = data_d[terms*k + 3];
                  const double D12 = data_d[terms*k + 4];
                  const double D22 = data_d[terms*k + 5];

                  const double q0 = data_qqq[0*quads + k];
                  const double q1 = data_qqq[1*quads + k];
                  const double q2 = data_qqq[2*quads + k];

                  data_qqq[0*quads + k] = D00 * q0 + D01 * q1 + D02 * q2;
                  data_qqq[1*quads + k] = D01 * q0 + D11 * q1 + D12 * q2;
                  data_qqq[2*quads + k] = D02 * q0 + D12 * q1 + D22 * q2;
               }

               // Apply transpose of the first operator that takes V -> QQQd -- QQQd -> U
               for (int k3 = 0; k3 < quads1d; ++k3)
               {
                  QQ = 0.;
                  for (int k2 = 0; k2 < quads1d; ++k2)
                  {
                     Q = 0.;
                     for (int k1 = 0; k1 < quads1d; ++k1)
                     {
                        const double q0 = QQQ0(k1, k2, k3);
                        const double q1 = QQQ1(k1, k2, k3);
                        const double q2 = QQQ2(k1, k2, k3);
                        for (int i1 = 0; i1 < dofs1d; ++i1)
                        {
                           Q(i1, 0) += q0 * dshape1d(i1, k1);
                           Q(i1, 1) += q1 * shape1d(i1, k1);
                           Q(i1, 2) += q2 * shape1d(i1, k1);
                        }
                     }
                     for (int i2 = 0; i2 < dofs1d; ++i2)
                     {
                        const double s = shape1d(i2, k2);
                        const double d = dshape1d(i2, k2);
                        for (int i1 = 0; i1 < dofs1d; ++i1)
                        {
                           QQ(i1, i2, 0) += Q(i1, 0) * s;
                           QQ(i1, i2, 1) += Q(i1, 1) * d;
                           QQ(i1, i2, 2) += Q(i1, 2) * s;
                        }
                     }
                  }
                  for (int i3 = 0; i3 < dofs1d; ++i3)
                  {
                     const double s = shape1d(i3, k3);
                     const double d = dshape1d(i3, k3);
                     for (int i2 = 0; i2 < dofs1d; ++i2)
                        for (int i1 = 0; i1 < dofs1d; ++i1)
                        {
                           Umat(i1, i2, i3) +=
                              QQ(i1, i2, 0) * s + 
                              QQ(i1, i2, 1) * s +
                              QQ(i1, i2, 2) * d;
                        }
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
      {
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
      {
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
      {
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
      }
      delete [] data_q;
      delete [] data_qq;
      delete [] data_qqq;
   }
}

}
