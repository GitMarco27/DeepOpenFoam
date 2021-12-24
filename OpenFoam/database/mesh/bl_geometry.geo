Include "bez_control_points.geo";
Include "params.geo";


Li  = 50.0;
Lo  = 50.0; // distance of inflow and outflow boundary from origin
Li1 = 2.0;
Lo1 = 20.0;   // distance of intermediate inflow and outflow from origin

Lce = 10;  // extrude size

n  = 200; // points on upper/lower surface of airfoil used to define airfoil

lc1 = 5.0;
lc2 = 0.00001; // characteristic lengths of elements on airfoil and at farfield
lc3 = 0.1; //characteristic length for the inetrmediate domain

m = 2*n - 2; // total number of points on airfoil without repetition
             // LE and TE points are common to upper/lower surface

nle = n; // point number of LE = no. of points on upper surface
         // Point(1) is trailing edge

Point(1) = {xs, ys, 0, lc2};
Point(2) = {x_u[1], y_u[1], 0, lc2};
Point(3) = {x_u[2], y_u[2], 0, lc2};
Point(4) = {x_u[3], y_u[3], 0, lc2};
Point(5) = {x_u[4], y_u[4], 0, lc2};
Point(6) = {x_u[5], y_u[5], 0, lc2};
Point(7) = {x_u[6], y_u[6], 0, lc2};
Point(8) = {x_u[7], y_u[7], 0, lc2};
Point(9) = {x_u[8], y_u[8], 0, lc2};
Point(10) = {x_u[9], y_u[9], 0, lc2};
Point(11) = {x_u[10], y_u[10], 0, lc2};
Point(12) = {xe, ye, 0, lc2};

Point(13) = {x_l[1], y_l[1], 0, lc2};
Point(14) = {x_l[2], y_l[2], 0, lc2};
Point(15) = {x_l[3], y_l[3], 0, lc2};
Point(16) = {x_l[4], y_l[4], 0, lc2};
Point(17) = {x_l[5], y_l[5], 0, lc2};
Point(18) = {x_l[6], y_l[6], 0, lc2};
Point(19) = {x_l[7], y_l[7], 0, lc2};
Point(20) = {x_l[8], y_l[8], 0, lc2};
Point(21) = {x_l[9], y_l[9], 0, lc2};
Point(22) = {x_l[10], y_l[10], 0, lc2};

Bezier(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
Bezier(2) = {12, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 1};

Transfinite Line{1,2} = n Using Bump 0.2;

Point(1001) = { 0.0, Li, 0.0,lc1};
Point(1002) = { 0.0, -Li, 0.0, lc1};
Point(1003) = {Lo, -Li, 0.0, lc1};
Point(1004) = {Lo, Li, 0.0, lc1};

Line(3) = {1004, 1001};
Circle(4) = {1001, 1, 1002};
Line(5) = {1002, 1003};
Line(6) = {1003, 1004};

Line Loop(1) = {1,2};
Line Loop(2) = {3,4,5,6};

//Intermediate Domain

Macro RotatePoint
    // rotates pointId about quarter-chord.
    // aoa is the angle of attack in degrees.
    Rotate {{0, 0, 1}, {0, 0, 0}, aoa * Pi / 180.0}
    {
        Point{pointId};
    }
Return

Point(2001) = { 0.0, Li1, 0.0,lc3};
pointId = 2001;
Call RotatePoint;
Point(2002) = { 0.0, -Li1, 0.0, lc3};
pointId = 2002;
Call RotatePoint;
Point(2003) = {Lo1, -Li1, 0.0, lc3};
pointId = 2003;
Call RotatePoint;
Point(2004) = {Lo1, Li1, 0.0, lc3};
pointId = 2004;
Call RotatePoint;



Line(13) = {2004, 2001};
Circle(14) = {2001, 1, 2002};
Line(15) = {2002, 2003};
Line(16) = {2003, 2004};

Line Loop(3) = {13,14,15,16};

Plane Surface(201) = {3,1};
Plane Surface(202) = {3,2};

Extrude {0,0,Lce} { Surface{201,202}; Layers{1}; Recombine;}

//Define Boundary Layer
Field[1] = BoundaryLayer;
Field[1].EdgesList = {1, 2};  // Tags of curves in the geometric model for which a boundary layer is needed
Field[1].AnisoMax = 1.0;  // Threshold angle for creating a mesh fan in the boundary layer
Field[1].FanNodesList = {12};  // Tags of points in the geometric model for which a fan is created
Field[1].FanPointsSizesList = {12};  // Number of elements in the fan for each fan node. If not present default value Mesh.BoundaryLayerFanElements
Field[1].hfar = 0.03;
Field[1].hwall_n = 0.00002;
Field[1].thickness = 0.05;  // Maximal thickness of the boundary layer
Field[1].ratio = 1.1;
Field[1].Quads = 1;
Field[1].IntersectMetrics = 0;  //  Intersect metrics of all surfaces
BoundaryLayer Field = 1;

Physical Surface("outlet") = {263, 275, 267};
Physical Surface("walls") = {};
Physical Surface("inlet") = {271};
// Physical Surface("airfoil") = {233, 229};
Physical Surface("p_side") = {229};
Physical Surface("s_side") = {233};
Physical Surface("back") = {276,234};
Physical Surface("front") = {201,202};
Physical Volume("volume") = {1, 2};//+
