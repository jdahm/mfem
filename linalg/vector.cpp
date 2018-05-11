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

// Implementation of data type vector

#include "vector.hpp"

#if defined(MFEM_USE_SUNDIALS) && defined(MFEM_USE_MPI)
#include <nvector/nvector_parallel.h>
#include <nvector/nvector_parhyp.h>
#endif

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>

namespace mfem
{

void Vector::Load(std::istream **in, int np, int *dim)
{
   int i, j, s;

   s = 0;
   for (i = 0; i < np; i++)
   {
      s += dim[i];
   }

   SetSize(s);

   double *data = array.GetData();
   int p = 0;
   for (i = 0; i < np; i++)
      for (j = 0; j < dim[i]; j++)
      {
         *in[i] >> data[p++];
      }
}

void Vector::Load(std::istream &in, int Size)
{
   SetSize(Size);

   for (int i = 0; i < Size; i++)
   {
      in >> array[i];
   }
}

double &Vector::Elem(int i)
{
   return operator()(i);
}

const double &Vector::Elem(int i) const
{
   return operator()(i);
}

double Vector::operator*(const double *v) const
{
   int s = array.Size();
   const double *d = array.GetData();
   double prod = 0.0;
   for (int i = 0; i < s; i++)
   {
      prod += d[i] * v[i];
   }
   return prod;
}

double Vector::operator*(const Vector &v) const
{
   MFEM_ASSERT(v.Size() == Size(), "Vector::operator*(const Vector &) const");

   return operator*(v.GetData());
}

Vector &Vector::operator=(const double *v)
{
   double *data = GetData();
   if (data != v)
   {
      MFEM_ASSERT(data + Size() <= v || v + Size() <= data, "Vectors overlap!");
      array.Assign(v);
   }
   return *this;
}

Vector &Vector::operator=(const Vector &v)
{
   SetSize(v.Size());
   return operator=(v.GetData());
}

Vector &Vector::operator=(double value)
{
   array = value;
   return *this;
}

Vector &Vector::operator*=(double c)
{
   for (int i = 0; i < Size(); i++)
   {
      array[i] *= c;
   }
   return *this;
}

Vector &Vector::operator/=(double c)
{
   const double m = 1.0/c;
   for (int i = 0; i < Size(); i++)
   {
      array[i] *= m;
   }
   return *this;
}

Vector &Vector::operator-=(double c)
{
   for (int i = 0; i < Size(); i++)
   {
      array[i] -= c;
   }
   return *this;
}

Vector &Vector::operator-=(const Vector &v)
{
   MFEM_ASSERT(Size() == v.Size(), "Vector::operator-=(const Vector &)");
   for (int i = 0; i < Size(); i++)
   {
      array[i] -= v(i);
   }
   return *this;
}

Vector &Vector::operator+=(const Vector &v)
{
   MFEM_ASSERT(Size() == v.Size(), "Vector::operator+=(const Vector &)");
   for (int i = 0; i < Size(); i++)
   {
      array[i] += v(i);
   }
   return *this;
}

Vector &Vector::Add(const double a, const Vector &Va)
{
   MFEM_ASSERT(Size() == Va.Size(), "Vector::Add(const double, const Vector &)");
   if (a != 0.0)
   {
      for (int i = 0; i < Size(); i++)
      {
         array[i] += a * Va(i);
      }
   }
   return *this;
}

Vector &Vector::Set(const double a, const Vector &Va)
{
   MFEM_ASSERT(Size() == Va.Size(), "Vector::Set(const double, const Vector &)");
   for (int i = 0; i < Size(); i++)
   {
      array[i] = a * Va(i);
   }
   return *this;
}

void Vector::SetVector(const Vector &v, int offset)
{
   const double *vp = v.GetData();
   double *p = GetData() + offset;
   MFEM_ASSERT(offset+v.Size() <= Size(), "Vector::SetVector(const Vector &, int)");
   std::memcpy(p, vp, sizeof(double)*v.Size());
}

void Vector::Neg()
{
   for (int i = 0; i < Size(); i++)
   {
      array[i] = -array[i];
   }
}

void add(const Vector &v1, const Vector &v2, Vector &v)
{
   const int vs = v.Size(), v1s = v1.Size(), v2s = v2.Size();
   const double *v1p = v1.GetData(), *v2p = v2.GetData();
   double *vp = v.GetData();
   MFEM_ASSERT((vs == v1s) && (vs == v2s), "add(Vector &v1, Vector &v2, Vector &v)");

   for (int i = 0; i < vs; i++)
   {
      vp[i] = v1p[i] + v2p[i];
   }
}

void add(const Vector &v1, double alpha, const Vector &v2, Vector &v)
{
   const int vs = v.Size(), v1s = v1.Size(), v2s = v2.Size();
   MFEM_ASSERT((vs == v1s) && (vs == v2s), "add(Vector &v1, double alpha, Vector &v2, Vector &v)");

   if (alpha == 0.0)
   {
      v = v1;
   }
   else if (alpha == 1.0)
   {
      add(v1, v2, v);
   }
   else
   {
      const double *v1p = v1.GetData(), *v2p = v2.GetData();
      double *vp = v.GetData();
      for (int i = 0; i < vs; i++)
      {
         vp[i] = v1p[i] + alpha*v2p[i];
      }
   }
}

void add(const double a, const Vector &x, const Vector &y, Vector &z)
{
   const int xs = x.Size(), ys = y.Size(), zs = z.Size();
   MFEM_ASSERT((xs == ys) && (xs == zs),
               "add(const double a, const Vector &x, const Vector &y,"
               " Vector &z)");
   if (a == 0.0)
   {
      z = 0.0;
   }
   else if (a == 1.0)
   {
      add(x, y, z);
   }
   else
   {
      const double *xp = x.GetData(), *yp = y.GetData();
      double *zp = z.GetData();

      for (int i = 0; i < zs; i++)
      {
         zp[i] = a * (xp[i] + yp[i]);
      }
   }
}

void add(const double a, const Vector &x,
         const double b, const Vector &y, Vector &z)
{
   const int xs = x.Size(), ys = y.Size(), zs = z.Size();
   MFEM_ASSERT((xs == ys) && (xs == zs),
               "add(const double a, const Vector &x,\n"
                 "    const double b, const Vector &y, Vector &z)");
   if (a == 0.0)
   {
      z.Set(b, y);
   }
   else if (b == 0.0)
   {
      z.Set(a, x);
   }
   else if (a == 1.0)
   {
      add(x, b, y, z);
   }
   else if (b == 1.0)
   {
      add(y, a, x, z);
   }
   else if (a == b)
   {
      add(a, x, y, z);
   }
   else
   {
      const double *xp = x.GetData(), *yp = y.GetData();
      double *zp = z.GetData();

      for (int i = 0; i < zs; i++)
      {
         zp[i] = a * xp[i] + b * yp[i];
      }
   }
}

void subtract(const Vector &x, const Vector &y, Vector &z)
{
   const int xs = x.Size(), ys = y.Size(), zs = z.Size();
   MFEM_ASSERT((xs == ys) && (xs == zs),
               "subtract(const Vector &, const Vector &, Vector &)");
   const double *xp = x.GetData(), *yp = y.GetData();
   double *zp = z.GetData();

   for (int i = 0; i < zs; i++)
   {
      zp[i] = xp[i] - yp[i];
   }
}

void subtract(const double a, const Vector &x, const Vector &y, Vector &z)
{
   const int xs = x.Size(), ys = y.Size(), zs = z.Size();
   MFEM_ASSERT((xs == ys) && (xs == zs),
               "subtract(const double a, const Vector &x,"
               " const Vector &y, Vector &z)");

   if (a == 0.)
   {
      z = 0.;
   }
   else if (a == 1.)
   {
      subtract(x, y, z);
   }
   else
   {
      const double *xp = x.GetData(), *yp = y.GetData();
      double *zp = z.GetData();

      for (int i = 0; i < zs; i++)
      {
         zp[i] = a * (xp[i] - yp[i]);
      }
   }
}

void Vector::median(const Vector &lo, const Vector &hi)
{
   for (int i = 0; i < Size(); i++)
   {
      if (array[i] < lo(i))
      {
         array[i] = lo(i);
      }
      else if (array[i] > hi(i))
      {
         array[i] = hi(i);
      }
   }
}

void Vector::GetSubVector(const Array<int> &dofs, Vector &elemvect) const
{
   elemvect.SetSize(dofs.Size());

   for (int i = 0; i < elemvect.Size(); i++)
   {
      const int j = dofs[i];
      elemvect(i) = (j >= 0) ? array[j] : -array[-1-j];
   }
}

void Vector::GetSubVector(const Array<int> &dofs, double *elem_data) const
{
   for (int i = 0; i < dofs.Size(); i++)
   {
      const int j = dofs[i];
      elem_data[i] = (j >= 0) ? array[j] : -array[-1-j];
   }
}

void Vector::SetSubVector(const Array<int> &dofs, const double value)
{
   for (int i = 0; i < dofs.Size(); i++)
   {
      const int j = dofs[i];
      if (j >= 0)
      {
         array[j] = value;
      }
      else
      {
         array[-1-j] = -value;
      }
   }
}

void Vector::SetSubVector(const Array<int> &dofs, const Vector &elemvect)
{
   for (int i = 0; i < dofs.Size(); i++)
   {
      const int j = dofs[i];
      if (j >= 0)
      {
         array[j] = elemvect(i);
      }
      else
      {
         array[-1-j] = -elemvect(i);
      }
   }
}

void Vector::SetSubVector(const Array<int> &dofs, double *elem_data)
{
   for (int i = 0; i < dofs.Size(); i++)
   {
      const int j = dofs[i];
      if (j >= 0)
      {
         array[j] = elem_data[i];
      }
      else
      {
         array[-1-j] = -elem_data[i];
      }
   }
}

void Vector::AddElementVector(const Array<int> &dofs, const Vector &elemvect)
{
   MFEM_ASSERT(dofs.Size() == elemvect.Size(), "");

   for (int i = 0; i < dofs.Size(); i++)
   {
      const int j = dofs[i];
      if (j >= 0)
      {
         array[j] += elemvect(i);
      }
      else
      {
         array[-1-j] -= elemvect(i);
      }
   }
}

void Vector::AddElementVector(const Array<int> &dofs, double *elem_data)
{
   for (int i = 0; i < dofs.Size(); i++)
   {
      const int j = dofs[i];
      if (j >= 0)
      {
         array[j] += elem_data[i];
      }
      else
      {
         array[-1-j] -= elem_data[i];
      }
   }
}

void Vector::AddElementVector(const Array<int> &dofs, const double a,
                              const Vector &elemvect)
{
   for (int i = 0; i < dofs.Size(); i++) {
      const int j = dofs[i];
      if (j >= 0)
      {
         array[j] += a * elemvect(i);
      }
      else
      {
         array[-1-j] -= a * elemvect(i);
      }
   }
}

void Vector::SetSubVectorComplement(const Array<int> &dofs, const double val)
{
   Vector dofs_vals;
   GetSubVector(dofs, dofs_vals);
   operator=(val);
   SetSubVector(dofs, dofs_vals);
}

void Vector::Print(std::ostream &out, int width) const
{
   const int size = Size();

   if (!size) { return; }

   for (int i = 0; 1; )
   {
      out << array[i];
      i++;
      if (i == size)
      {
         break;
      }
      if ( i % width == 0 )
      {
         out << '\n';
      }
      else
      {
         out << ' ';
      }
   }
   out << '\n';
}

void Vector::Print_HYPRE(std::ostream &out) const
{
   std::ios::fmtflags old_fmt = out.flags();
   out.setf(std::ios::scientific);
   std::streamsize old_prec = out.precision(14);

   out << Size() << '\n';  // number of rows

   for (int i = 0; i < Size(); i++)
   {
      out << array[i] << '\n';
   }

   out.precision(old_prec);
   out.flags(old_fmt);
}

void Vector::Randomize(int seed)
{
   // static unsigned int seed = time(0);
   const double max = (double)(RAND_MAX) + 1.;

   if (seed == 0)
   {
      seed = (int)time(0);
   }

   // srand(seed++);
   srand((unsigned)seed);

   for (int i = 0; i < Size(); i++)
   {
      array[i] = std::abs(rand()/max);
   }
}

double Vector::Norml2() const
{
   const int size = Size();
   // Scale entries of Vector on the fly, using algorithms from
   // std::hypot() and LAPACK's drm2. This scaling ensures that the
   // argument of each call to std::pow is <= 1 to avoid overflow.
   if (0 == size)
   {
      return 0.0;
   } // end if 0 == size

   if (1 == size)
   {
      return std::abs(array[0]);
   } // end if 1 == size

   double scale = 0.0;
   double sum = 0.0;

   for (int i = 0; i < size; i++)
   {
      if (array[i] != 0.0)
      {
         const double absdata = std::abs(array[i]);
         if (scale <= absdata)
         {
            const double sqr_arg = scale / absdata;
            sum = 1.0 + sum * (sqr_arg * sqr_arg);
            scale = absdata;
            continue;
         } // end if scale <= absdata
         const double sqr_arg = absdata / scale;
         sum += (sqr_arg * sqr_arg); // else scale > absdata
      } // end if data[i] != 0
   }
   return scale * std::sqrt(sum);
}

double Vector::Normlinf() const
{
   double max = 0.0;
   for (int i = 0; i < Size(); i++)
   {
      max = std::max(std::abs(array[i]), max);
   }
   return max;
}

double Vector::Norml1() const
{
   double sum = 0.0;
   for (int i = 0; i < Size(); i++)
   {
      sum += std::abs(array[i]);
   }
   return sum;
}

double Vector::Normlp(double p) const
{
   MFEM_ASSERT(p > 0.0, "Vector::Normlp");
   const int size = Size();

   if (p == 1.0)
   {
      return Norml1();
   }
   if (p == 2.0)
   {
      return Norml2();
   }
   if (p < infinity())
   {
      // Scale entries of Vector on the fly, using algorithms from
      // std::hypot() and LAPACK's drm2. This scaling ensures that the
      // argument of each call to std::pow is <= 1 to avoid overflow.
      if (0 == size)
      {
         return 0.0;
      } // end if 0 == size

      if (1 == size)
      {
         return std::abs(array[0]);
      } // end if 1 == size

      double scale = 0.0;
      double sum = 0.0;

      for (int i = 0; i < size; i++)
      {
         if (array[i] != 0.0)
         {
            const double absdata = std::abs(array[i]);
            if (scale <= absdata)
            {
               sum = 1.0 + sum * std::pow(scale / absdata, p);
               scale = absdata;
               continue;
            } // end if scale <= absdata
            sum += std::pow(absdata / scale, p); // else scale > absdata
         } // end if data[i] != 0
      }
      return scale * std::pow(sum, 1.0/p);
   } // end if p < infinity()

   return Normlinf(); // else p >= infinity()
}

double Vector::Max() const
{
   double max = array[0];

   for (int i = 1; i < Size(); i++)
      if (array[i] > max)
      {
         max = array[i];
      }

   return max;
}

double Vector::Min() const
{
   double min = array[0];

   for (int i = 1; i < Size(); i++)
      if (array[i] < min)
      {
         min = array[i];
      }

   return min;
}

double Vector::Sum() const
{
   double sum = 0.0;

   for (int i = 0; i < Size(); i++)
   {
      sum += array[i];
   }

   return sum;
}

#ifdef MFEM_USE_SUNDIALS

#ifndef SUNTRUE
#define SUNTRUE TRUE
#endif
#ifndef SUNFALSE
#define SUNFALSE FALSE
#endif

Vector::Vector(N_Vector nv)
{
   N_Vector_ID nvid = N_VGetVectorID(nv);
   switch (nvid)
   {
      case SUNDIALS_NVEC_SERIAL:
         SetDataAndSize(NV_DATA_S(nv), NV_LENGTH_S(nv));
         break;
#ifdef MFEM_USE_MPI
      case SUNDIALS_NVEC_PARALLEL:
         SetDataAndSize(NV_DATA_P(nv), NV_LOCLENGTH_P(nv));
         break;
      case SUNDIALS_NVEC_PARHYP:
      {
         hypre_Vector *hpv_local = N_VGetVector_ParHyp(nv)->local_vector;
         SetDataAndSize(hpv_local->data, hpv_local->size);
         break;
      }
#endif
      default:
         MFEM_ABORT("N_Vector type " << nvid << " is not supported");
   }
}

void Vector::ToNVector(N_Vector &nv)
{
   MFEM_ASSERT(nv, "N_Vector handle is NULL");
   N_Vector_ID nvid = N_VGetVectorID(nv);
   switch (nvid)
   {
      case SUNDIALS_NVEC_SERIAL:
         MFEM_ASSERT(NV_OWN_DATA_S(nv) == SUNFALSE, "invalid serial N_Vector");
         NV_DATA_S(nv) = data;
         NV_LENGTH_S(nv) = size;
         break;
#ifdef MFEM_USE_MPI
      case SUNDIALS_NVEC_PARALLEL:
         MFEM_ASSERT(NV_OWN_DATA_P(nv) == SUNFALSE, "invalid parallel N_Vector");
         NV_DATA_P(nv) = data;
         NV_LOCLENGTH_P(nv) = size;
         break;
      case SUNDIALS_NVEC_PARHYP:
      {
         hypre_Vector *hpv_local = N_VGetVector_ParHyp(nv)->local_vector;
         MFEM_ASSERT(hpv_local->owns_data == false, "invalid hypre N_Vector");
         hpv_local->data = data;
         hpv_local->size = size;
         break;
      }
#endif
      default:
         MFEM_ABORT("N_Vector type " << nvid << " is not supported");
   }
}

#endif // MFEM_USE_SUNDIALS

}
