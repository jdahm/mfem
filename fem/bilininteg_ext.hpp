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

#ifndef MFEM_BILININTEG_EXT
#define MFEM_BILININTEG_EXT

namespace mfem
{

// ***************************************************************************
// * GeometryExtension
// ***************************************************************************
class GeometryExtension
{
public:
   Array<int> eMap;
   Array<double> nodes;
   Array<double> X, J, invJ, detJ;
   static const int Jacobian    = (1 << 0);
   static const int JacobianInv = (1 << 1);
   static const int JacobianDet = (1 << 2);
   static GeometryExtension* Get(const FiniteElementSpace&,
                                 const IntegrationRule&);
   static GeometryExtension* Get(const FiniteElementSpace&,
                                 const IntegrationRule&,
                                 const Vector&);
   static void ReorderByVDim(const GridFunction*);
   static void ReorderByNodes(const GridFunction*);
};

// ***************************************************************************
// * DofToQuad
// ***************************************************************************
class DofToQuad
{
private:
   std::string hash;
public:
   Array<double> W, B, G, Bt, Gt;
public:
   static void delDofToQuad();
   static DofToQuad* Get(const mfem::FiniteElementSpace&,
                         const mfem::IntegrationRule&,
                         const bool = false);
   static DofToQuad* Get(const mfem::FiniteElementSpace&,
                         const mfem::FiniteElementSpace&,
                         const mfem::IntegrationRule&,
                         const bool = false);
   static DofToQuad* Get(const mfem::FiniteElement&,
                         const mfem::FiniteElement&,
                         const mfem::IntegrationRule&,
                         const bool = false);
   static DofToQuad* GetTensorMaps(const mfem::FiniteElement&,
                                   const mfem::FiniteElement&,
                                   const mfem::IntegrationRule&,
                                   const bool = false);
   static DofToQuad* GetD2QTensorMaps(const mfem::FiniteElement&,
                                      const mfem::IntegrationRule&,
                                      const bool = false);
   static DofToQuad* GetSimplexMaps(const mfem::FiniteElement&,
                                    const mfem::IntegrationRule&,
                                    const bool = false);
   static DofToQuad* GetSimplexMaps(const mfem::FiniteElement&,
                                    const mfem::FiniteElement&,
                                    const mfem::IntegrationRule&,
                                    const bool = false);
   static DofToQuad* GetD2QSimplexMaps(const mfem::FiniteElement&,
                                       const mfem::IntegrationRule&,
                                       const bool = false);
};

}

#endif
