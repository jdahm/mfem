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

extern bool skip_target;

struct DeviceSpec
{
   enum Class
   {
      HOST = 1,
      ACCEL = 2
   };

   enum Class type;
   int id;

   DeviceSpec() : type(HOST), id(0) { }
   bool UseTarget() const { return (type == ACCEL) && (!skip_target); }
};

void SetDefaultAccelerator(int id);
void UseHost();

} // namespace mfem

#endif
