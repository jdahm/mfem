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


// Device class for storing the execution device information.

#include "device.hpp"

#if defined(MFEM_USE_OPENMP)
#include <omp.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace mfem
{

static int numHostCores()
{
#if !defined(_WIN32)
   return sysconf(_SC_NPROCESSORS_ONLN);
#else
   SYSTEM_INFO sysinfo;
   GetSystemInfo(&sysinfo);
   return = sysinfo.dwNumberOfProcessors;
#endif
}

static int numHostThreads()
{
   int num_threads = 1;
#if defined(MFEM_USE_OPENMP)
#pragma omp parallel
   {
#pragma omp master
      {
         num_threads = omp_get_num_threads();
      }
   }
#endif
   return num_threads;
}

Device::Device() :
   classification(HOST),
   num_cores(numHostCores()),
   num_threads(numHostThreads()),
   accel_id(-1) { }

void Device::SetAccelerator(const int _accel_id)
{
#if defined(MFEM_USE_OPENMP)
#pragma omp single
   {
      classification = ACCEL;
      accel_id = _accel_id;
      omp_set_default_device(accel_id);
   }
#endif
}

bool Device::Target() const
{
   return classification == ACCEL;
}

Device ExecDevice;

} // namespace mfem
