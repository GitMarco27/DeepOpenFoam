Include "naca.geo";
Include "WindTunnel.geo";
Include "parameters.geo";

// Units are multiples of chord.

ce = 0;
Point(ce++) = {0, 0, 0};

Call SymmetricAirfoil;

// Creating the wind tunnel

WindTunnelHeight = 20;
WindTunnelLength = 40;
WindTunnelLc = 1;
Call WindTunnel;

// Creating surfaces

Surface(ce++) = {WindTunnelLoop, AirfoilLoop};
TwoDimSurf = ce - 1;
Recombine Surface{TwoDimSurf};

ids[] = Extrude {0, 0, 0.1}
{
	Surface{TwoDimSurf};
	Layers{1};
	Recombine;
};

Physical Surface("outlet") = {ids[2]};
Physical Surface("walls") = {ids[{3, 5}]};
Physical Surface("inlet") = {ids[4]};
Physical Surface("airfoil") = {ids[{6:8}]};
Physical Surface("frontAndBack") = {ids[0], TwoDimSurf};
Physical Volume("volume") = {ids[1]};

