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

#ifndef MFEM_DEVICE
#define MFEM_DEVICE

#include "../config/config.hpp"

namespace mfem
{

class Device
{
public:
   // The Device Class is meant to guide the choice of suitable computational
   // kernels. A kernel may be classified as being suitable for a combination of
   // device classes.
   enum Class
   {
      HOST = 1,
      ACCEL = 2
   };

   Device();

   void SetAccelerator(const int _accel_id = 0);

   bool UsesTarget() const;

protected:
   int classification;
   int num_cores;
   int num_threads;
   int accel_id;
};

extern Device ExecDevice;

} // namespace mfem

#endif
