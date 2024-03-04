% Define the center and radius of the circle
center = [0, 0];
radius = 1;

% Generate the x and y coordinates of the circle
x_inf = [-1, -1, 1, 1, -1];
y_inf = [-1, 1, 1, -1, -1];

x_1 = [-1, 0, 1, 0, -1];
y_1 = [0, 1, 0, -1, 0];

x_1_larger = sqrt(2) .* x_1;
y_1_larger = sqrt(2) .* y_1;

x_1_llarger = 2 .* x_1;
y_1_llarger = 2 .* y_1;

theta = linspace(0, 2*pi, 100);
x_2 = center(1) + radius * cos(theta);
y_2 = center(2) + radius * sin(theta);

radius = 2/sqrt(pi);
x_2_larger = center(1) + radius * cos(theta);
y_2_larger = center(2) + radius * sin(theta);

radius = sqrt(2);
x_2_llarger = center(1) + radius * cos(theta);
y_2_llarger = center(2) + radius * sin(theta);

% Plot the circle
plot(x_inf, y_inf, '-');
hold on;
plot(x_1, y_1, '-');
hold on;
plot(x_2, y_2, '-');
hold on;
plot(x_1_larger, y_1_larger, '--');
hold on;
plot(x_2_larger, y_2_larger, '--');
hold on;
plot(x_1_llarger, y_1_llarger, '-.');
hold on;
plot(x_2_llarger, y_2_llarger, '-.');
axis equal;
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
ax.Box = 'off';
mycolors = [1 0 0; 0 1 0; 0 0 1; 0 1 0; 0 0 1; 0 1 0; 0 0 1];
ax = gca; 
ax.ColorOrder = mycolors;
%lgd = legend('$\{x : \|x\|_1 = 1\}$','$\{x : \|x\|_2 = 1\}$','$\{x : \|x\|_\infty = 1\}$','$\{x : \|x\|_1 = \sqrt{2}\}$','$\{x : \|x\|_2 = 2/\sqrt{\pi}\}$','Interpreter','latex')
lgd = legend('$\varepsilon_\infty =1$','$\varepsilon_1 = 1$','$\varepsilon_2 =1$','$\varepsilon_1 = \sqrt{2}$','$\varepsilon_2 = 2/\sqrt{\pi}$','$\varepsilon_1 = 2$','$\varepsilon_2 = \sqrt{2}$','Interpreter','latex');
lgd.FontSize = 12;

% Set the ticks on both axes equal to 0.5 steps up to 1.5
ax.XTick = -2:0.5:2;
ax.YTick = -2:0.5:2;