steps = [1,2,4,8,16];

%NATURAL
%constant
constant_rand_top1 = [0.09, 0.008, 0.002, 0.002, 0.0];
constant_det_top1 = [0.088, 0.006, 0.002, 0.002, 0.0];

constant_rand_top5 = [0.412, 0.154, 0.092, 0.082, 0.072];
constant_det_top5 = [0.402, 0.136, 0.082, 0.068, 0.068];

%harmonic
harmonic_rand_top1 = [0.09, 0.008, 0.002, 0.002, 0.002];
harmonic_det_top1 = [0.088, 0.006, 0.002, 0.002, 0.002];

harmonic_rand_top5 = [0.412, 0.142, 0.078, 0.07, 0.062];
harmonic_det_top5 = [0.402, 0.13, 0.066, 0.064, 0.058];

%geometric
geometric_rand_top1 = [0.09, 0.008, 0.002, 0.002, 0.002];
geometric_det_top1 = [0.088, 0.006, 0.002, 0.002, 0.002];

geometric_rand_top5 = [0.412, 0.142, 0.076, 0.068, 0.072];
geometric_det_top5 = [0.402, 0.13, 0.07, 0.06, 0.06];

Ndif1 = max([max(constant_rand_top1 - harmonic_rand_top1), max(constant_rand_top1 - geometric_rand_top1), max(harmonic_rand_top1 - geometric_rand_top1)])
Ndif2 = max([max(constant_det_top1 - harmonic_det_top1), max(constant_det_top1 - geometric_det_top1), max(harmonic_det_top1 - geometric_det_top1)])
%Ndif3 = 0;
%Ndif4 = 0;
Ndif3 = max([max(constant_rand_top5 - harmonic_rand_top5), max(constant_rand_top5 - geometric_rand_top5), max(harmonic_rand_top5 - geometric_rand_top5)])
Ndif4 = max([max(constant_det_top5 - harmonic_det_top5), max(constant_det_top5 - geometric_det_top5), max(harmonic_det_top5 - geometric_det_top5)])

plot(steps, constant_det_top5, "r-", steps, constant_rand_top5, "r--", steps, harmonic_det_top5, "g-", steps, harmonic_rand_top5, "g--", steps, geometric_det_top5, "b-", steps, geometric_rand_top5, "b--");
legend('const. det.', 'const. rand.', 'harm. det.', 'harm. rand.', 'geom. det.', 'geom. rand.');
xticks(steps)
ylabel('Top-5 Accuracy')
xlabel('Number of Steps')
xlim([1 16])
ylim([0 0.5])

%plot(steps, constant_det_top1, "r-", steps, constant_rand_top1, "r--", steps, harmonic_det_top1, "g-", steps, harmonic_rand_top1, "g--", steps, geometric_det_top1, "b-", steps, geometric_rand_top1, "b--");
%legend('const. det.', 'const. rand.', 'harm. det.', 'harm. rand.', 'geom. det.', 'geom. rand.');
%xticks(steps)
%ylabel('Top-1 Accuracy')
%xlabel('Number of Steps')

%MADRY
%constant
constant_rand_top1 = [0.742, 0.712, 0.708, 0.704, 0.702];
constant_det_top1 = [0.706, 0.676, 0.672, 0.67, 0.666];

constant_rand_top5 = [0.976, 0.974, 0.972, 0.97, 0.97];
constant_det_top5 = [0.97, 0.966, 0.958, 0.958, 0.952];

%harmonic
harmonic_rand_top1 = [0.742, 0.714, 0.708, 0.708, 0.71];
harmonic_det_top1 = [0.706, 0.682, 0.674, 0.672, 0.67];

harmonic_rand_top5 = [0.976, 0.974, 0.972, 0.972, 0.97];
harmonic_det_top5 = [0.97, 0.968, 0.96, 0.958, 0.958];

%geometric
geometric_rand_top1 = [0.742, 0.714, 0.708, 0.708, 0.708];
geometric_det_top1 = [0.706, 0.682, 0.674, 0.674, 0.674];

geometric_rand_top5 = [0.976, 0.974, 0.972, 0.972, 0.972];
geometric_det_top5 = [0.97, 0.968, 0.96, 0.96, 0.96];

Mdif1 = max([max(constant_rand_top1 - harmonic_rand_top1), max(constant_rand_top1 - geometric_rand_top1), max(harmonic_rand_top1 - geometric_rand_top1)])
Mdif2 = max([max(constant_det_top1 - harmonic_det_top1), max(constant_det_top1 - geometric_det_top1), max(harmonic_det_top1 - geometric_det_top1)])
Mdif3 = max([max(constant_rand_top5 - harmonic_rand_top5), max(constant_rand_top5 - geometric_rand_top5), max(harmonic_rand_top5 - geometric_rand_top5)])
Mdif4 = max([max(constant_det_top5 - harmonic_det_top5), max(constant_det_top5 - geometric_det_top5), max(harmonic_det_top5 - geometric_det_top5)])


%LOCUS
%constant
constant_rand_top1 = [0.62, 0.61, 0.604, 0.602, 0.6];
constant_det_top1 = [0.596, 0.58, 0.574, 0.57, 0.566];

constant_rand_top5 = [0.928, 0.916, 0.914, 0.916, 0.916];
constant_det_top5 = [0.912, 0.908, 0.902, 0.904, 0.904];

%harmonic
harmonic_rand_top1 = [0.62, 0.61, 0.606, 0.604, 0.602];
harmonic_det_top1 = [0.596, 0.584, 0.574, 0.574, 0.572];

harmonic_rand_top5 = [0.928, 0.916, 0.916, 0.916, 0.916];
harmonic_det_top5 = [0.912, 0.908, 0.902, 0.902, 0.902];

%geometric
geometric_rand_top1 = [0.62, 0.61, 0.606, 0.606, 0.606];
geometric_det_top1 = [0.596, 0.584, 0.578, 0.574, 0.574];

geometric_rand_top5 = [0.928, 0.916, 0.916, 0.916, 0.916];
geometric_det_top5 = [0.912, 0.908, 0.904, 0.904, 0.904];

Ldif1 = max([max(constant_rand_top1 - harmonic_rand_top1), max(constant_rand_top1 - geometric_rand_top1), max(harmonic_rand_top1 - geometric_rand_top1)])
Ldif2 = max([max(constant_det_top1 - harmonic_det_top1), max(constant_det_top1 - geometric_det_top1), max(harmonic_det_top1 - geometric_det_top1)])
Ldif3 = max([max(constant_rand_top5 - harmonic_rand_top5), max(constant_rand_top5 - geometric_rand_top5), max(harmonic_rand_top5 - geometric_rand_top5)])
Ldif4 = max([max(constant_det_top5 - harmonic_det_top5), max(constant_det_top5 - geometric_det_top5), max(harmonic_det_top5 - geometric_det_top5)])

Dif = max([Ldif1, Ldif2, Ldif3, Ldif4, Mdif1, Mdif2, Mdif3, Mdif4, Ndif1, Ndif2, Ndif3, Ndif4])