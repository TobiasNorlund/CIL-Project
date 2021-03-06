
load ALS
load SGD
load EMNMF
load WNMF
close all

SGD(3,:) = (1:size(SGD,2))*40;
ALS(3,:) = (1:size(ALS,2))*7;
EMNMF(3,:) = (1:size(EMNMF,2))*2;
WNMF(3,:) = (1:size(WNMF,2))*2;

ALS(3,end) = SGD(3,end);
EMNMF(3,end) = SGD(3,end);
WNMF(3,end) = SGD(3,end);

semilogx(SGD(3,:), SGD(2,:), '-k', 'LineWidth',2 )
hold on
semilogx(SGD(3,:), SGD(1,:), '--k')

semilogx(ALS(3,:), ALS(1,:), '-b', 'LineWidth',2 )
semilogx(ALS(3,:), ALS(2,:), '--b')

semilogx(EMNMF(3,:), EMNMF(1,:), '-g', 'LineWidth',2 )
semilogx(EMNMF(3,:), EMNMF(2,:), '--g')

semilogx(WNMF(3,:), WNMF(1,:), '-r', 'LineWidth',2 )
semilogx(WNMF(3,:), WNMF(2,:), '--r')
hold off

axis([-inf, 2000, 0.8, 1.3])
legend('SGD (test)', 'SGD (train)', 'ALS (test)', 'ALS (train)', 'EMNMF (test)', 'EMNMF (train)', 'WNMF (test)', 'WNMF (train)')
xlabel('Time (s)')
ylabel('RMSE')