numeric: float
vars: x,y
semiring: minplus
lp:
minimize max(x,y-4);
max(x,y) <=3;
x >= 1;
y >= 1;

basic point = 3,1;

