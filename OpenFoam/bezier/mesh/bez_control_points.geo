// u: upper, l: lower, s: start, e: end;

xs = 0;
ys = 0;
xe = 1;
ye = 0;

// startind and ending point must be fixed
// the first control point must have x = 0 and y must be the same for pressure and suction side
//

x_u = {xs, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, xe};
x_l = {xs, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, xe};
y_u = {ys, 0.05, 0.06, 0.07, 0.08, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, ye};
y_l = {ys, -0.05, -0.08, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.05, ye};
