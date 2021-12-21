Include "bez_control_points.geo";
Include "fixed_params.geo";

ce=0;
//+
ObjectLines[] ={};

Point(ce++) = {xs, ys, 0, ObjectLc};
StartPoint = ce - 1;
Point(ce++) = {ux1, uy1, 0, ObjectLc};
Point(ce++) = {ux2, uy2, 0, ObjectLc};
Point(ce++) = {ux3, uy3, 0, ObjectLc};
Point(ce++) = {xe, ye, 0, ObjectLc};
EndPoint = ce - 1;
Bezier(ce++) = {StartPoint, ce-5, ce-4, ce-3, EndPoint};
// Transfinite Curve{ce-1} = nPoints;
ObjectLines += ce - 1;
Point(ce++) = {lx1, ly1, 0, ObjectLc};
Point(ce++) = {lx2, ly2, 0, ObjectLc};
Point(ce++) = {lx3, ly3, 0, ObjectLc};
Bezier(ce++) = {EndPoint, ce-2, ce-3, ce-4, StartPoint};
// Transfinite Curve{ce-1} = nPoints;
ObjectLines += ce - 1;

Line Loop(ce++) = ObjectLines[];
ObjectLoop = ce - 1;

//+
WindTunnelLines[] ={};
Point(ce++) = {WindTunnelLength / 2, -WindTunnelHeight / 2, 0, WindTunnelLc};
StartPoint = ce - 1;
Point(ce++) = {WindTunnelLength / 2, WindTunnelHeight / 2, 0, WindTunnelLc};
Line(ce++) = {ce - 3, ce - 2};
WindTunnelLines += ce - 1;

Point(ce++) = {-WindTunnelLength / 2, WindTunnelHeight / 2, 0, WindTunnelLc};
Line(ce++) = {ce - 4, ce - 2};
WindTunnelLines += ce - 1;

Point(ce++) = {-WindTunnelLength / 2, -WindTunnelHeight / 2, 0, WindTunnelLc};
Line(ce++) = {ce - 4, ce - 2};
WindTunnelLines += ce - 1;

Line(ce++) = {ce - 3, StartPoint};
WindTunnelLines += ce - 1;

Line Loop(ce++) = WindTunnelLines[];
WindTunnelLoop = ce - 1;

//+
Surface(ce++) = {WindTunnelLoop, ObjectLoop};
TwoDimSurf = ce - 1;
Recombine Surface{TwoDimSurf};

//+
Field[1] = BoundaryLayer;
Field[1].CurvesList = {ObjectLines[0], ObjectLines[1]};
//Field[1].hfar = 0.0005;
//Field[1].hwall_n = 0.0000005;
//Field[1].thickness = 0.0001; //0.0004
//Field[1].ratio = 1.5;
//Field[1].Quads = 1;
BoundaryLayer Field = 1;

ids[] = Extrude {0, 0, 0.1}
{
	Surface{TwoDimSurf};
	Layers{1};
	Recombine;
};

Physical Surface("outlet") = {ids[2]};
Physical Surface("walls") = {ids[{3, 5}]};
Physical Surface("inlet") = {ids[4]};
Physical Surface("airfoil") = {ids[{6, 7}]};
Physical Surface("frontAndBack") = {ids[0], TwoDimSurf};
Physical Volume("volume") = {ids[1]};
//+
Physical Surface("airfoil", 4) += {51, 47};
