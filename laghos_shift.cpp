// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "laghos_shift.hpp"
#include "laghos_solver.hpp"

using namespace std;
namespace mfem
{

namespace hydrodynamics
{

int material_id(int el_id, const ParGridFunction &g)
{
   const ParFiniteElementSpace &pfes =  *g.ParFESpace();
   const FiniteElement *fe = pfes.GetFE(el_id);
   Vector g_vals;
   const IntegrationRule &ir =
      IntRules.Get(fe->GetGeomType(), pfes.GetOrder(el_id) + 7);

   double integral = 0.0;
   bool is_positive = true;
   g.GetValues(el_id, ir, g_vals);
   ElementTransformation *Tr = pfes.GetMesh()->GetElementTransformation(el_id);
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      integral += ip.weight * g_vals(q) * Tr->Weight();
      if (g_vals(q) < 0.0) { is_positive = false; }
   }
   return (is_positive) ? 1 : 0;
   //return (integral > 0.0) ? 1 : 0;
}

double interfaceLS(const Vector &x)
{
   const int mode = 1;

   const int dim = x.Size();
   switch (mode)
   {
      case 0: return tanh(x(0) - 0.5);
      case 1:
      {
      double center[3] = {0.5, 0.5, 0.5};
         double rad = 0.0;
         for (int d = 0; d < dim; d++)
         {
            rad += (x(d) - center[d]) * (x(d) - center[d]);
         }
         rad = sqrt(rad + 1e-16);
         return tanh(rad - 0.3);
      }
      default: MFEM_ABORT("error"); return 0.0;
   }
}

void MarkFaceAttributes(ParMesh &pmesh)
{
   // Set face_attribute = 77 to faces that are on the material interface.
   for (int f = 0; f < pmesh.GetNumFaces(); f++)
   {
      auto *ftr = pmesh.GetFaceElementTransformations(f, 3);
      if (ftr->Elem2No > 0 &&
          pmesh.GetAttribute(ftr->Elem1No) != pmesh.GetAttribute(ftr->Elem2No))
      {
         pmesh.SetFaceAttribute(f, 77);
      }
   }
}

void FaceForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                                             const FiniteElement &test_fe1,
                                             const FiniteElement &test_fe2,
                                             FaceElementTransformations &Trans,
                                             DenseMatrix &elmat)
{
   const int h1dofs_cnt_face = trial_face_fe.GetDof();
   const int l2dofs_cnt = test_fe1.GetDof();
   const int dim = test_fe1.GetDim();

   if (Trans.Elem2No < 0)
   {
      // This case should take care of shared (MPI) faces. They will get
      // processed by both MPI tasks.
      elmat.SetSize(l2dofs_cnt, h1dofs_cnt_face * dim);
   }
   elmat.SetSize(l2dofs_cnt * 2, h1dofs_cnt_face * dim);
   elmat = 0.0;

   if (Trans.Attribute != 77) { return; }

   h1_shape_face.SetSize(h1dofs_cnt_face);
   l2_shape.SetSize(l2dofs_cnt);

   const int ir_order =
      test_fe1.GetOrder() + trial_face_fe.GetOrder() + Trans.OrderW();
   const IntegrationRule *ir = &IntRules.Get(Trans.GetGeometryType(), ir_order);
   const int nqp_face = ir->GetNPoints();

   // d at all face quad points.

   // grad_p at all quad points, on both sides.
   Vector p_e;
   Array<int> dofs_p;
   const FiniteElement &el_p = *p.ParFESpace()->GetFE(0);
   const int dof_p = el_p.GetDof();
   DenseMatrix p_grad_e_1(dof_p, dim), p_grad_e_2(dof_p, dim);
   DenseMatrix grad_phys; // This will be (dof_p x dim, dof_p).
   {
      p.ParFESpace()->GetElementDofs(Trans.Elem1No, dofs_p);
      p.GetSubVector(dofs_p, p_e);
      ElementTransformation &Tr_el1 = Trans.GetElement1Transformation();
      el_p.ProjectGrad(el_p, Tr_el1, grad_phys);
      Vector grad_ptr(p_grad_e_1.GetData(), dof_p*dim);
      grad_phys.Mult(p_e, grad_ptr);
   }
   if (Trans.Elem2No > 0)
   {
      p.ParFESpace()->GetElementDofs(Trans.Elem1No, dofs_p);
      p.GetSubVector(dofs_p, p_e);
      ElementTransformation &Tr_el2 = Trans.GetElement2Transformation();
      el_p.ProjectGrad(el_p, Tr_el2, grad_phys);
      Vector grad_ptr(p_grad_e_2.GetData(), dof_p*dim);
      grad_phys.Mult(p_e, grad_ptr);
   }

   Vector nor(dim);

   Vector p_grad_q(dim), d_q(dim), shape_p(dof_p);
   for (int q = 0; q  < nqp_face; q++)
   {
      const IntegrationPoint &ip_f = ir->IntPoint(q);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip_f);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &ip_e1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &ip_e2 = Trans.GetElement2IntPoint();

      // The normal includes the Jac scaling.
      // The orientation is taken into account in the processing of element 1.
      if (dim == 1)
      {
         nor(0) = (2*ip_e1.x - 1.0 ) * Trans.Weight();
      }
      else { CalcOrtho(Trans.Jacobian(), nor); }
      nor *= ip_f.weight;

      // Shape functions on the face (H1); same for both elements.
      trial_face_fe.CalcShape(ip_f, h1_shape_face);

      // 1st element.
      {
         // Compute dist * grad_p in the first element.
         el_p.CalcShape(ip_e1, shape_p);
         p_grad_e_1.MultTranspose(shape_p, p_grad_q);
         dist.Eval(d_q, Trans.GetElement1Transformation(), ip_e1);
         const double grad_p_d = d_q * p_grad_q;

         // The normal must be outward w.r.t. element 1.
         // For attr 2, we always have d.n_2 > 0.
         // For attr 1, we always have d.n_1 < 0.
         if (Trans.GetElement1Transformation().Attribute == 2)
         {
            if (nor * d_q < 0.0) { nor *= -1.0; MFEM_ABORT("test2"); }
         }
         else
         {
            if (nor * d_q > 0.0) { nor *= -1.0; MFEM_ABORT("test1"); }
         }

         // L2 shape functions in the 1st element.
         test_fe1.CalcShape(ip_e1, l2_shape);

         for (int i = 0; i < l2dofs_cnt; i++)
         {
            for (int j = 0; j < h1dofs_cnt_face; j++)
            {
               for (int d = 0; d < dim; d++)
               {
                  elmat(i, d*h1dofs_cnt_face + j) +=
                        grad_p_d * l2_shape(i) * h1_shape_face(j) * nor(d);
               }
            }
         }
      }
      // 2nd element if there is such (subtracting from the 1st).
      if (Trans.Elem2No >= 0)
      {
         // Compute dist * grad_p in the second element.
         el_p.CalcShape(ip_e2, shape_p);
         p_grad_e_2.MultTranspose(shape_p, p_grad_q);
         dist.Eval(d_q, Trans.GetElement2Transformation(), ip_e2);
         const double grad_p_d = d_q * p_grad_q;

         // L2 shape functions on the 2nd element.
         test_fe2.CalcShape(ip_e2, l2_shape);
         for (int i = 0; i < l2dofs_cnt; i++)
         {
            for (int j = 0; j < h1dofs_cnt_face; j++)
            {
               for (int d = 0; d < dim; d++)
               {
                  elmat(l2dofs_cnt + i, d*h1dofs_cnt_face + j) -=
                        grad_p_d * l2_shape(i) * h1_shape_face(j) * nor(d);
               }
            }
         }
      }
   }
}


} // namespace hydrodynamics

} // namespace mfem
