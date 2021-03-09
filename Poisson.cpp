#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    double dx = x[0] - 0.5;
    double dy = x[1] - 0.5;
    values[0] =  10*exp(-((dx + 0.3)*(dx + 0.3) + (dy - 0.3)*(dy - 0.3))*100) + 10*exp(-((dx - 0.3)*(dx -0.3) + (dy - 0.3)*(dy - 0.3))*100) +10*exp(-((dx)*(dx) + (dy)*(dy))*90);
  }
};

// Normal derivative (Neumann boundary condition)
class dUdN : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = (x[0]*x[0] + x[1]*x[1]);
  }

};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return abs(x[0]) < DOLFIN_EPS or abs(x[1]) < DOLFIN_EPS or x[0] > 1 - DOLFIN_EPS or x[1] > 1 - DOLFIN_EPS;
  }
};

int main()
{
  // Create mesh and function space
  auto mesh = std::make_shared<Mesh>(UnitSquareMesh::create({{64, 64}}, CellType::Type::triangle));
  auto V = std::make_shared<Poisson::FunctionSpace>(mesh);

  // Define boundary condition
  auto u0 = std::make_shared<Constant>(0.0);
  auto boundary = std::make_shared<DirichletBoundary>();
  DirichletBC bc(V, u0, boundary);

  // Define variational forms
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  auto f = std::make_shared<Source>();
  auto g = std::make_shared<dUdN>();
  L.f = f;
  L.g = g;

  // Compute solution
  Function u(V);
  solve(a == L, u, bc);

  // Save solution in VTK format
  File file("poisson.pvd");
  file << u;

  return 0;
}
