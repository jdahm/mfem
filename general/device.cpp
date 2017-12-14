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

#include "device.hpp"

#if defined(MFEM_USE_OPENMP)
#include <omp.h>
#endif

namespace mfem
{

bool skip_target = false;

void SetDefaultAccelerator(int id)
{
#if defined(MFEM_USE_OPENMP)
     omp_set_default_device(id);
#endif
}

void UseHost() { skip_target = true; }


} // namespace mfem

#endif
