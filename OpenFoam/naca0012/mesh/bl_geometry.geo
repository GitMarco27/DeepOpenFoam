Li  = 50.0;
Lo  = 50.0; // distance of inflow and outflow boundary from origin
Li1 = 2.0;
Lo1 = 2.0;   // distance of intermediate inflow and outflow from origin

n  = 201; // points on upper/lower surface of airfoil used to define airfoil
// These points may not appear in the mesh.

lc1 = 10.0;
lc2 = 0.00001; // characteristic lengths of elements on airfoil and at farfield
lc3 = 0.1; //characteristic length for the inetrmediate domain
m = 2*n - 2; // total number of points on airfoil without repetition
             // LE and TE points are common to upper/lower surface

nle = n; // point number of LE = no. of points on upper surface
         // Point(1) is trailing edge

Macro NACA0012
   x2 = x * x;
   x3 = x * x2;
   x4 = x * x3;
   y = 0.594689181*(0.298222773*Sqrt(x)
       - 0.127125232*x - 0.357907906*x2 + 0.291984971*x3 - 0.105174606*x4);
Return

// put points on upper surface of airfoil
For i In {1:n}
   theta = Pi * (i-1) / (n-1);
   x = 0.5 * (Cos(theta) + 1.0);
   Call NACA0012;
   Point(i) = {x, y, 0.0, lc2};
   xx[i] = x;
   yy[i] = y;
EndFor

// put points on lower surface of airfoil, use upper surface points and reflect
For i In {n+1:m}
   Point(i) = {xx[2*n-i], -yy[2*n-i], 0.0, lc2};
EndFor

Spline(1) = {1:n}; Spline(2) = {n:m,1};

Transfinite Line{1,2} = n Using Bump 0.2;

Point(1001) = { 0.0, Li, 0.0,lc1};
Point(1002) = { 0.0, -Li, 0.0, lc1};
Point(1003) = {Lo, -Li, 0.0, lc1};
Point(1004) = {Lo, Li, 0.0, lc1};

Line(3) = {1004, 1001};
Circle(4) = {1001, nle, 1002};
Line(5) = {1002, 1003};
Line(6) = {1003, 1004};

Line Loop(1) = {1,2};
Line Loop(2) = {3,4,5,6};

//Intermediate Domain
Point(2001) = { 0.0, Li1, 0.0,lc3};
Point(2002) = { 0.0, -Li1, 0.0, lc3};
Point(2003) = {Lo1+10, -Li1, 0.0, lc3};
Point(2004) = {Lo1+10, Li1, 0.0, lc3};

Line(13) = {2004, 2001};
Circle(14) = {2001, nle, 2002};
Line(15) = {2002, 2003};
Line(16) = {2003, 2004};

Line Loop(3) = {13,14,15,16};

Plane Surface(201) = {3,1};
Plane Surface(202) = {3,2};

Extrude {0,0,lc1} { Surface{201,202}; Layers{1}; Recombine;}

//Define Boundary Layer
Field[1] = BoundaryLayer;
Field[1].EdgesList = {1,2};
Field[1].AnisoMax = 1.0;
Field[1].FanNodesList = {1};
Field[1].FanPointsSizesList = {12};
Field[1].hfar = 0.03;
Field[1].hwall_n = 0.00002;
Field[1].thickness = 0.05;
Field[1].ratio = 1.1;
Field[1].Quads = 1;
Field[1].IntersectMetrics = 0;
BoundaryLayer Field = 1;

// Physical Surface(201) = {201,202};  //Periodic at z = 0
// Physical Surface(202) = {276, 234}; //Periodic at z = lc1
// Physical Surface(203) = {271, 275, 263, 267}; //Farfield
// Physical Volume(1)    = {1, 2};

Physical Surface("outlet") = {263, 275, 267};
Physical Surface("walls") = {};
Physical Surface("inlet") = {271};
Physical Surface("airfoil") = {233, 229};
Physical Surface("back") = {276,234};
Physical Surface("front") = {201,202};
Physical Volume("volume") = {1, 2};//+
