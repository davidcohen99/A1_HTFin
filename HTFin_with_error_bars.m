% Code for HTFin Lab

% Break down given dataset A by medium
% Note: Cycle 1 ends at scan 2010 and Cycle 2 starts at scan 2140

load('HTFin_Cycle1_and_Cycle2.mat')

% Put data on graph to see what we're looking for
figure('Name','Data Visualization')
subplot(2,2,1)
sgtitle('HTFin Data Visualization')
plot(1:size(A,1),A(:,1))
hold on
plot(1:size(A,1),A(:,2))
plot(1:size(A,1),A(:,3))
plot(1:size(A,1),A(:,4))
plot(1:size(A,1),A(:,5))
title('Brass')
hold off

subplot(2,2,2)
plot(1:size(A,1),A(:,6))
hold on
plot(1:size(A,1),A(:,7))
plot(1:size(A,1),A(:,8))
plot(1:size(A,1),A(:,9))
plot(1:size(A,1),A(:,10))
title('Copper')
hold off

subplot(2,2,3)
plot(1:size(A,1),A(:,11))
hold on
plot(1:size(A,1),A(:,12))
plot(1:size(A,1),A(:,13))
plot(1:size(A,1),A(:,14))
plot(1:size(A,1),A(:,15))
title('Steel')
hold off

subplot(2,2,4)
plot(1:size(A,1),A(:,16))
hold on
plot(1:size(A,1),A(:,17))
plot(1:size(A,1),A(:,18))
plot(1:size(A,1),A(:,19))
plot(1:size(A,1),A(:,20))
title('Aluminum')
hold off

%% Free convection

brass1 = A(1:2010,1:5);
copper1 = A(1:2010,6:10);
steel2 = A(2140:end,11:15);
aluminum2 = A(2140:end,16:20);

% Figure 1 
figure('Name','Brass - Free Convection')
% subplot(2,2,1)
% sgtitle('Figure 1: Free Convection')

%plot(1:size(brass1,1),brass1(:,5))
errorbar(1:50:size(brass1,1),brass1(1:50:end,5), ... 
         0.5*ones(1,ceil((size(brass1,1)/50))))
hold on

%plot(1:size(brass1,1),brass1(:,4))
errorbar(1:50:size(brass1,1),brass1(1:50:end,4), ... 
         0.5*ones(1,ceil((size(brass1,1)/50))))

%plot(1:size(brass1,1),brass1(:,3))
errorbar(1:50:size(brass1,1),brass1(1:50:end,3), ... 
         0.5*ones(1,ceil((size(brass1,1)/50))))
     
%plot(1:size(brass1,1),brass1(:,2))
errorbar(1:50:size(brass1,1),brass1(1:50:end,2), ... 
         0.5*ones(1,ceil((size(brass1,1)/50))))
     
%plot(1:size(brass1,1),brass1(:,1))
errorbar(1:50:size(brass1,1),brass1(1:50:end,1), ... 
         0.5*ones(1,ceil((size(brass1,1)/50))))
     
xline(1710,'k--');
title('Brass - Free Convection')
xlabel('Time (scans)')
ylabel('Temperature (Celsius)')
legend('5','4','3','2','1','SS','Location','Southeast')
hold off

% subplot(2,2,2)
figure('Name','Copper - Free Convection')
%plot(1:size(copper1,1),copper1(:,5))
errorbar(1:50:size(copper1,1),copper1(1:50:end,5), ... 
         0.5*ones(1,ceil((size(copper1,1)/50))))
hold on
%plot(1:size(copper1,1),copper1(:,4))
errorbar(1:50:size(copper1,1),copper1(1:50:end,4), ... 
         0.5*ones(1,ceil((size(copper1,1)/50))))
     
%plot(1:size(copper1,1),copper1(:,3))
errorbar(1:50:size(copper1,1),copper1(1:50:end,3), ... 
         0.5*ones(1,ceil((size(copper1,1)/50))))
     
%plot(1:size(copper1,1),copper1(:,2))
errorbar(1:50:size(copper1,1),copper1(1:50:end,2), ... 
         0.5*ones(1,ceil((size(copper1,1)/50))))
     
%plot(1:size(copper1,1),copper1(:,1))
errorbar(1:50:size(copper1,1),copper1(1:50:end,1), ... 
         0.5*ones(1,ceil((size(copper1,1)/50))))
     
title('Copper - Free Convection')
xline(1950,'k--');
xlabel('Time (scans)')
ylabel('Temperature (Celsius)')
legend('5','4','3','2','1','SS','Location','Southeast')
hold off

% subplot(2,2,3)
figure('Name','Steel - Free Convection')
%plot(1:size(steel2,1),steel2(:,5))
errorbar(1:50:size(steel2,1),steel2(1:50:end,5), ... 
         0.5*ones(1,ceil((size(steel2,1)/50))))
hold on
%plot(1:size(steel2,1),steel2(:,4))
errorbar(1:50:size(steel2,1),steel2(1:50:end,4), ... 
         0.5*ones(1,ceil((size(steel2,1)/50))))
     
%plot(1:size(steel2,1),steel2(:,3))
errorbar(1:50:size(steel2,1),steel2(1:50:end,3), ... 
         0.5*ones(1,ceil((size(steel2,1)/50))))
     
%plot(1:size(steel2,1),steel2(:,2))
errorbar(1:50:size(steel2,1),steel2(1:50:end,2), ... 
         0.5*ones(1,ceil((size(steel2,1)/50))))
     
%plot(1:size(steel2,1),steel2(:,1))
errorbar(1:50:size(steel2,1),steel2(1:50:end,1), ... 
         0.5*ones(1,ceil((size(steel2,1)/50))))
     
xline(1400,'k--');
title('Steel - Free Convection')
xlabel('Time (scans)')
ylabel('Temperature (Celsius)')
legend('5','4','3','2','1','SS','Location','east')
hold off

% subplot(2,2,4)
figure('Name','Aluminum - Free Convection')
%plot(1:size(aluminum2,1),aluminum2(:,5))
errorbar(1:50:size(aluminum2,1),aluminum2(1:50:end,5), ... 
         0.5*ones(1,ceil((size(aluminum2,1)/50))))
hold on
%plot(1:size(aluminum2,1),aluminum2(:,4))
errorbar(1:50:size(aluminum2,1),aluminum2(1:50:end,4), ... 
         0.5*ones(1,ceil((size(aluminum2,1)/50))))
     
%plot(1:size(aluminum2,1),aluminum2(:,3))
errorbar(1:50:size(aluminum2,1),aluminum2(1:50:end,3), ... 
         0.5*ones(1,ceil((size(aluminum2,1)/50))))
     
%plot(1:size(aluminum2,1),aluminum2(:,2))
errorbar(1:50:size(aluminum2,1),aluminum2(1:50:end,2), ... 
         0.5*ones(1,ceil((size(aluminum2,1)/50))))
     
%plot(1:size(aluminum2,1),aluminum2(:,1))
errorbar(1:50:size(aluminum2,1),aluminum2(1:50:end,1), ... 
         0.5*ones(1,ceil((size(aluminum2,1)/50))))
     
xline(1380,'k--');
title('Aluminum - Free Convection')
xlabel('Time (scans)')
ylabel('Temperature (Celsius)')
legend('5','4','3','2','1','SS','Location','Southeast')
hold off

% Temperatures at steady state
table1SSfree = [brass1(1710,1) brass1(1710,2) brass1(1710,3) brass1(1710,4) brass1(1710,5);
                copper1(1950,1) copper1(1950,2) copper1(1950,3) copper1(1950,4) copper1(1950,5);
                steel2(1400,1) steel2(1400,2) steel2(1400,3) steel2(1400,4) steel2(1400,5);
                aluminum2(1380,1) aluminum2(1380,2) aluminum2(1380,3) aluminum2(1380,4) aluminum2(1380,5)]
    
%% Forced convections

brass2 = A(2140:end,1:5);
copper2 = A(2140:end,6:10);
steel1 = A(1:2010,11:15);
aluminum1 = A(1:2010,16:20);

% Figure 2
% figure('Name','Forced Convection')
% subplot(2,2,1)
% sgtitle('Figure 2: Forced Convection')

figure('Name','Brass - Forced Convection')
%plot(1:size(brass2,1),brass2(:,5))
errorbar(1:50:size(brass2,1),brass2(1:50:end,5), ... 
         0.5*ones(1,ceil((size(brass2,1)/50))))
hold on
%plot(1:size(brass2,1),brass2(:,4))
errorbar(1:50:size(brass2,1),brass2(1:50:end,4), ... 
         0.5*ones(1,ceil((size(brass2,1)/50))))
     
%plot(1:size(brass2,1),brass2(:,3))
errorbar(1:50:size(brass2,1),brass2(1:50:end,3), ... 
         0.5*ones(1,ceil((size(brass2,1)/50))))
     
%plot(1:size(brass2,1),brass2(:,2))
errorbar(1:50:size(brass2,1),brass2(1:50:end,2), ... 
         0.5*ones(1,ceil((size(brass2,1)/50))))
     
%plot(1:size(brass2,1),brass2(:,1))
errorbar(1:50:size(brass2,1),brass2(1:50:end,1), ... 
         0.5*ones(1,ceil((size(brass2,1)/50))))
     
xline(470,'k--');
title('Brass - Forced Convection')
xlabel('Time (scans)')
ylabel('Temperature (Celsius)')
legend('5','4','3','2','1','SS','Location','northeast')
hold off

figure('Name','Copper - Forced Convection')
% subplot(2,2,2)
%plot(1:size(copper2,1),copper2(:,5))
errorbar(1:50:size(copper2,1),copper2(1:50:end,5), ... 
         0.5*ones(1,ceil((size(copper2,1)/50))))
hold on
%plot(1:size(copper2,1),copper2(:,4))
errorbar(1:50:size(copper2,1),copper2(1:50:end,4), ... 
         0.5*ones(1,ceil((size(copper2,1)/50))))
     
%plot(1:size(copper2,1),copper2(:,3))
errorbar(1:50:size(copper2,1),copper2(1:50:end,3), ... 
         0.5*ones(1,ceil((size(copper2,1)/50))))
     
%plot(1:size(copper2,1),copper2(:,2))
errorbar(1:50:size(copper2,1),copper2(1:50:end,2), ... 
         0.5*ones(1,ceil((size(copper2,1)/50))))
     
%plot(1:size(copper2,1),copper2(:,1))
errorbar(1:50:size(copper2,1),copper2(1:50:end,1), ... 
         0.5*ones(1,ceil((size(copper2,1)/50))))
     
xline(750,'k--');
title('Copper - Forced Convection')
xlabel('Time (scans)')
ylabel('Temperature (Celsius)')
legend('5','4','3','2','1','SS','Location','northeast')
hold off

figure('Name','Steel - Forced Convection')
% subplot(2,2,3)
%plot(1:size(steel1,1),steel1(:,5))
errorbar(1:50:size(steel1,1),steel1(1:50:end,5), ... 
         0.5*ones(1,ceil((size(steel1,1)/50))))
hold on
%plot(1:size(steel1,1),steel1(:,4))
errorbar(1:50:size(steel1,1),steel1(1:50:end,4), ... 
         0.5*ones(1,ceil((size(steel1,1)/50))))
     
%plot(1:size(steel1,1),steel1(:,3))
errorbar(1:50:size(steel1,1),steel1(1:50:end,3), ... 
         0.5*ones(1,ceil((size(steel1,1)/50))))
     
%plot(1:size(steel1,1),steel1(:,2))
errorbar(1:50:size(steel1,1),steel1(1:50:end,2), ... 
         0.5*ones(1,ceil((size(steel1,1)/50))))
     
%plot(1:size(steel1,1),steel1(:,1))
errorbar(1:50:size(steel1,1),steel1(1:50:end,1), ... 
         0.5*ones(1,ceil((size(steel1,1)/50))))
     
xline(1800,'k--');
title('Steel - Forced Convection')
xlabel('Time (scans)')
ylabel('Temperature (Celsius)')
legend('5','4','3','2','1','SS','Location','Southeast')
hold off

figure('Name','Aluminum - Forced Convection')
% subplot(2,2,4)
%plot(1:size(aluminum1,1),aluminum1(:,5))
errorbar(1:50:size(aluminum2,1),aluminum2(1:50:end,5), ... 
         0.5*ones(1,ceil((size(aluminum2,1)/50))))
hold on
%plot(1:size(aluminum1,1),aluminum1(:,4))
errorbar(1:50:size(aluminum2,1),aluminum2(1:50:end,4), ... 
         0.5*ones(1,ceil((size(aluminum2,1)/50))))
     
%plot(1:size(aluminum1,1),aluminum1(:,3))
errorbar(1:50:size(aluminum2,1),aluminum2(1:50:end,3), ... 
         0.5*ones(1,ceil((size(aluminum2,1)/50))))
     
%plot(1:size(aluminum1,1),aluminum1(:,2))
errorbar(1:50:size(aluminum2,1),aluminum2(1:50:end,2), ... 
         0.5*ones(1,ceil((size(aluminum2,1)/50))))
     
%plot(1:size(aluminum1,1),aluminum1(:,1))
errorbar(1:50:size(aluminum2,1),aluminum2(1:50:end,1), ... 
         0.5*ones(1,ceil((size(aluminum2,1)/50))))
     
xline(1600,'k--');
title('Aluminum - Forced Convection')
xlabel('Time (scans)')
ylabel('Temperature (Celsius)')
legend('5','4','3','2','1','SS','Location','Southeast')
hold off

% Temperatures at steady state
table1SSforced = [brass2(470,1) brass2(470,2) brass2(470,3) brass2(470,4) brass2(470,5);
                copper2(750,1) copper2(750,2) copper2(750,3) copper2(750,4) copper2(750,5);
                steel1(1800,1) steel1(1800,2) steel1(1800,3) steel1(1800,4) steel1(1800,5);
                aluminum1(1600,1) aluminum1(1600,2) aluminum1(1600,3) aluminum1(1600,4) aluminum1(1600,5)]
    
%% Least Squares Approx.

% Find room temperature
T_inf = mean(A(1,:))
L = 0.3048; % m
xdist = [0.002 0.0762 0.1524 0.2286 0.3028]; % m
k = [118.19 399.60 15.03 237.18]
d = 0.0128; % m
P = pi()*d;
a = (pi()*d.^2)./4;
% m = sqrt((h*P)./(k*a));

brass = [brass1(1710,5) brass1(1710,4) brass1(1710,3) brass1(1710,2) brass1(1710,1);
        brass2(470,5) brass2(470,4) brass2(470,3) brass2(470,2) brass2(470,1)];
copper = [copper1(1950,5) copper1(1950,4) copper1(1950,3) copper1(1950,2) copper1(1950,1);
        copper2(750,5) copper2(750,4) copper2(750,3) copper2(750,2) copper2(750,1)];
steel = [steel2(1400,5) steel2(1400,4) steel2(1400,3) steel2(1400,2) steel2(1400,1);
        steel1(1800,5) steel1(1800,4) steel1(1800,3) steel1(1800,2) steel1(1800,1)];
alum = [aluminum2(1380,5) aluminum2(1380,4) aluminum2(1380,3) aluminum2(1380,2) aluminum2(1380,1);
        aluminum1(1600,5) aluminum1(1600,4) aluminum1(1600,3) aluminum1(1600,2) aluminum1(1600,1)];

T_exact_br_fr = @(h,x) T_inf + (brass(1,1)- T_inf).*((cosh(sqrt((h.*P)./(k(1).*a))*(L-x)) + (h./((sqrt((h.*P)./(k(1).*a)).*k(1)))).*sinh(sqrt((h.*P)./(k(1).*a)).*(L-x)))./(cosh(sqrt((h.*P)./(k(1).*a)).*L) +(h./((sqrt((h.*P)./(k(1).*a))*k(1)))).*sinh(sqrt((h.*P)./(k(1).*a))*L)));
T_exact_br_fo = @(h,x) T_inf + (brass(2,1)- T_inf).*((cosh(sqrt((h.*P)./(k(1).*a))*(L-x)) + (h./((sqrt((h.*P)./(k(1).*a)).*k(1)))).*sinh(sqrt((h.*P)./(k(1).*a)).*(L-x)))./(cosh(sqrt((h.*P)./(k(1).*a)).*L) +(h./((sqrt((h.*P)./(k(1).*a))*k(1)))).*sinh(sqrt((h.*P)./(k(1).*a))*L)));
h_br_fr = lsqcurvefit(T_exact_br_fr, 5, xdist, brass(1,1:end));
h_br_fo = lsqcurvefit(T_exact_br_fo, 20, xdist, brass(2,1:end));

T_exact_cu_fr = @(h,x) T_inf + (copper(1,1)- T_inf).*((cosh(sqrt((h.*P)./(k(2).*a))*(L-x)) + (h./((sqrt((h.*P)./(k(2).*a)).*k(2)))).*sinh(sqrt((h.*P)./(k(2).*a)).*(L-x)))./(cosh(sqrt((h.*P)./(k(2).*a)).*L) +(h./((sqrt((h.*P)./(k(2).*a))*k(2)))).*sinh(sqrt((h.*P)./(k(2).*a))*L)));
T_exact_cu_fo = @(h,x) T_inf + (copper(2,1)- T_inf).*((cosh(sqrt((h.*P)./(k(2).*a))*(L-x)) + (h./((sqrt((h.*P)./(k(2).*a)).*k(2)))).*sinh(sqrt((h.*P)./(k(2).*a)).*(L-x)))./(cosh(sqrt((h.*P)./(k(2).*a)).*L) +(h./((sqrt((h.*P)./(k(2).*a))*k(2)))).*sinh(sqrt((h.*P)./(k(2).*a))*L)));
h_cu_fr = lsqcurvefit(T_exact_cu_fr, 5, xdist, copper(1,1:end));
h_cu_fo = lsqcurvefit(T_exact_cu_fo, 20, xdist, copper(2,1:end));

T_exact_st_fr = @(h,x) T_inf + (steel(1,1)- T_inf).*((cosh(sqrt((h.*P)./(k(3).*a)).*(L-x)) + (h./((sqrt((h.*P)./(k(3).*a)).*k(3)))).*sinh(sqrt((h.*P)./(k(3).*a)).*(L-x)))./(cosh(sqrt((h.*P)./(k(3).*a)).*L) +(h./((sqrt((h.*P)./(k(3).*a)).*k(3)))).*sinh(sqrt((h.*P)./(k(3).*a)).*L)));
T_exact_st_fo = @(h,x) T_inf + (steel(2,1)- T_inf).*((cosh(sqrt((h.*P)./(k(3).*a)).*(L-x)) + (h./((sqrt((h.*P)./(k(3).*a)).*k(3)))).*sinh(sqrt((h.*P)./(k(3).*a)).*(L-x)))./(cosh(sqrt((h.*P)./(k(3).*a)).*L) +(h./((sqrt((h.*P)./(k(3).*a)).*k(3)))).*sinh(sqrt((h.*P)./(k(3).*a)).*L)));
h_st_fr = lsqcurvefit(T_exact_st_fr, 5, xdist, steel(1,1:end));
h_st_fo = lsqcurvefit(T_exact_st_fo, 20, xdist, steel(2,1:end));

T_exact_al_fr = @(h,x) T_inf + (alum(1,1)- T_inf)*((cosh(sqrt((h*P)./(k(4)*a))*(L-x)) + (h./((sqrt((h*P)./(k(4)*a))*k(4))))*sinh(sqrt((h*P)./(k(4)*a))*(L-x)))./(cosh(sqrt((h*P)./(k(4)*a))*L) +(h./((sqrt((h*P)./(k(4)*a))*k(4))))*sinh(sqrt((h*P)./(k(4)*a))*L)));
T_exact_al_fo = @(h,x) T_inf + (alum(2,1)- T_inf)*((cosh(sqrt((h*P)./(k(4)*a))*(L-x)) + (h./((sqrt((h*P)./(k(4)*a))*k(4))))*sinh(sqrt((h*P)./(k(4)*a))*(L-x)))./(cosh(sqrt((h*P)./(k(4)*a))*L) +(h./((sqrt((h*P)./(k(4)*a))*k(4))))*sinh(sqrt((h*P)./(k(4)*a))*L)));
h_al_fr = lsqcurvefit(T_exact_al_fr, 5, xdist, alum(1,1:end));
h_al_fo = lsqcurvefit(T_exact_al_fo, 20, xdist, alum(2,1:end));

h = [h_br_fr h_cu_fr h_st_fr h_al_fr;
    h_br_fo h_cu_fo h_st_fo h_al_fo] % top row free, bottom row forced. order: brass copper steel aluminum

%% Figure 3 - Free Convection
T_exact_fr = [];
for i=1:5
    T_exact_fr(1,i) = T_inf + (brass(1,1)- T_inf).*((cosh(sqrt((h_br_fr.*P)./(k(1).*a))*(L-xdist(i))) + (h_br_fr./((sqrt((h_br_fr.*P)./(k(1).*a)).*k(1)))).*sinh(sqrt((h_br_fr.*P)./(k(1).*a)).*(L-xdist(i))))./(cosh(sqrt((h_br_fr.*P)./(k(1).*a)).*L) +(h_br_fr./((sqrt((h_br_fr.*P)./(k(1).*a))*k(1)))).*sinh(sqrt((h_br_fr.*P)./(k(1).*a))*L)));
    T_exact_fr(2,i) = T_inf + (copper(1,1)- T_inf).*((cosh(sqrt((h_cu_fr.*P)./(k(2).*a))*(L-xdist(i))) + (h_cu_fr./((sqrt((h_cu_fr.*P)./(k(2).*a)).*k(2)))).*sinh(sqrt((h_cu_fr.*P)./(k(2).*a)).*(L-xdist(i))))./(cosh(sqrt((h_cu_fr.*P)./(k(2).*a)).*L) +(h_cu_fr./((sqrt((h_cu_fr.*P)./(k(2).*a))*k(2)))).*sinh(sqrt((h_cu_fr.*P)./(k(2).*a))*L)));
    T_exact_fr(3,i) = T_inf + (steel(1,1)- T_inf).*((cosh(sqrt((h_st_fr.*P)./(k(3).*a)).*(L-xdist(i))) + (h_st_fr./((sqrt((h_st_fr.*P)./(k(3).*a)).*k(3)))).*sinh(sqrt((h_st_fr.*P)./(k(3).*a)).*(L-xdist(i))))./(cosh(sqrt((h_st_fr.*P)./(k(3).*a)).*L) +(h_st_fr./((sqrt((h_st_fr.*P)./(k(3).*a)).*k(3)))).*sinh(sqrt((h_st_fr.*P)./(k(3).*a)).*L)));
    T_exact_fr(4,i) = T_inf + (alum(1,1)- T_inf).*((cosh(sqrt((h_al_fr.*P)./(k(4).*a))*(L-xdist(i))) + (h_al_fr./((sqrt((h_al_fr.*P)./(k(4).*a)).*k(4))))*sinh(sqrt((h_al_fr*P)./(k(4)*a))*(L-xdist(i))))./(cosh(sqrt((h_al_fr*P)./(k(4)*a))*L)+(h_al_fr./((sqrt((h_al_fr*P)./(k(4)*a))*k(4))))*sinh(sqrt((h_al_fr*P)./(k(4)*a))*L)));
end 
figure('Name','Temp Distribution vs Fin Length (Free Convection)')
% plot(xdist,T_exact_fr(1,:),'b')
fplot(@(x) T_inf + (brass(1,1)- T_inf).*((cosh(sqrt((h(1,1).*P)./(k(1).*a))*(L-x)) + (h(1,1)./((sqrt((h(1,1).*P)./(k(1).*a)).*k(1)))).*sinh(sqrt((h(1,1).*P)./(k(1).*a)).*(L-x)))./(cosh(sqrt((h(1,1).*P)./(k(1).*a)).*L) +(h(1,1)./((sqrt((h(1,1).*P)./(k(1).*a))*k(1)))).*sinh(sqrt((h(1,1).*P)./(k(1).*a))*L))),'b')
hold on
errorbar(xdist,brass(1,:),0.5*ones(1,numel(xdist)), 'b.','HandleVisibility','off')

% plot(xdist,T_exact_fr(2,:),'r')
fplot(@(x) T_inf + (copper(1,1)- T_inf).*((cosh(sqrt((h(1,2).*P)./(k(2).*a))*(L-x)) + (h(1,2)./((sqrt((h(1,2).*P)./(k(2).*a)).*k(2)))).*sinh(sqrt((h(1,2).*P)./(k(2).*a)).*(L-x)))./(cosh(sqrt((h(1,2).*P)./(k(2).*a)).*L) +(h(1,2)./((sqrt((h(1,2).*P)./(k(2).*a))*k(2)))).*sinh(sqrt((h(1,2).*P)./(k(2).*a))*L))),'r')
errorbar(xdist,copper(1,:),0.5*ones(1,numel(xdist)), 'r.','HandleVisibility','off')

% plot(xdist,T_exact_fr(3,:),'m')
fplot(@(x) T_inf + (steel(1,1)- T_inf).*((cosh(sqrt((h(1,3).*P)./(k(3).*a)).*(L-x)) + (h(1,3)./((sqrt((h(1,3).*P)./(k(3).*a)).*k(3)))).*sinh(sqrt((h(1,3).*P)./(k(3).*a)).*(L-x)))./(cosh(sqrt((h(1,3).*P)./(k(3).*a)).*L) +(h(1,3)./((sqrt((h(1,3).*P)./(k(3).*a))*k(3)))).*sinh(sqrt((h(1,3).*P)./(k(3).*a))*L))),'m')
errorbar(xdist,steel(1,:),0.5*ones(1,numel(xdist)), 'm.','HandleVisibility','off')

% plot(xdist,T_exact_fr(4,:),'g')
fplot(@(x) T_inf + (alum(1,1)- T_inf).*((cosh(sqrt((h(1,4).*P)./(k(4).*a)).*(L-x)) + (h(1,4)./((sqrt((h(1,4).*P)./(k(4).*a)).*k(4)))).*sinh(sqrt((h(1,4).*P)./(k(4).*a)).*(L-x)))./(cosh(sqrt((h(1,4).*P)./(k(4).*a)).*L) +(h(1,4)./((sqrt((h(1,4).*P)./(k(4).*a))*k(4)))).*sinh(sqrt((h(1,4).*P)./(k(4).*a))*L))),'g')
errorbar(xdist,alum(1,:),0.5*ones(1,numel(xdist)), 'g.','HandleVisibility','off')

xlabel('Fin Length (m)')
ylabel('Temperature (Celsius)')
title('Temperature Distribution vs Fin Length (Free Convection)')
legend('Brass','Copper','Steel','Aluminum')
xlim([0 L])
ylim([0 85])
hold off

%% Figure 4 - Forced Convection

T_exact_fo = [];
for i=1:5
    T_exact_fo(1,i) = T_inf + (brass(2,1)- T_inf).*((cosh(sqrt((h_br_fo.*P)./(k(1).*a)).*(L-xdist(i))) + (h_br_fo./((sqrt((h_br_fo.*P)./(k(1).*a)).*k(1)))).*sinh(sqrt((h_br_fo.*P)./(k(1).*a)).*(L-xdist(i))))./(cosh(sqrt((h_br_fo.*P)./(k(1).*a)).*L) +(h_br_fo./((sqrt((h_br_fo.*P)./(k(1).*a))*k(1)))).*sinh(sqrt((h_br_fo.*P)./(k(1).*a))*L)));
    T_exact_fo(2,i) = T_inf + (copper(2,1)- T_inf).*((cosh(sqrt((h_cu_fo.*P)./(k(2).*a)).*(L-xdist(i))) + (h_cu_fo./((sqrt((h_cu_fo.*P)./(k(2).*a)).*k(2)))).*sinh(sqrt((h_cu_fo.*P)./(k(2).*a)).*(L-xdist(i))))./(cosh(sqrt((h_cu_fo.*P)./(k(2).*a)).*L) +(h_cu_fo./((sqrt((h_cu_fo.*P)./(k(2).*a))*k(2)))).*sinh(sqrt((h_cu_fo.*P)./(k(2).*a))*L)));
    T_exact_fo(3,i) = T_inf + (steel(2,1)- T_inf).*((cosh(sqrt((h_st_fo.*P)./(k(3).*a)).*(L-xdist(i))) + (h_st_fo./((sqrt((h_st_fo.*P)./(k(3).*a)).*k(3)))).*sinh(sqrt((h_st_fo.*P)./(k(3).*a)).*(L-xdist(i))))./(cosh(sqrt((h_st_fo.*P)./(k(3).*a)).*L) +(h_st_fo./((sqrt((h_st_fo.*P)./(k(3).*a)).*k(3)))).*sinh(sqrt((h_st_fo.*P)./(k(3).*a)).*L)));
    T_exact_fo(4,i) = T_inf + (alum(2,1)- T_inf).*((cosh(sqrt((h_al_fo.*P)./(k(4).*a)).*(L-xdist(i))) + (h_al_fo./((sqrt((h_al_fo.*P)./(k(4).*a)).*k(4)))).*sinh(sqrt((h_al_fo.*P)./(k(4).*a)).*(L-xdist(i))))./(cosh(sqrt((h_al_fo.*P)./(k(4).*a)).*L) +(h_al_fo./((sqrt((h_al_fo.*P)./(k(4).*a)).*k(4)))).*sinh(sqrt((h_al_fo.*P)./(k(4).*a)).*L)));
end 
figure('Name','Temp Distribution vs Fin Length (Forced Convection)')
% plot(xdist,T_exact_fo(1,:), 'b')
fplot(@(x) T_inf + (brass(2,1)- T_inf).*((cosh(sqrt((h(2,1).*P)./(k(1).*a))*(L-x)) + (h(2,1)./((sqrt((h(2,1).*P)./(k(1).*a)).*k(1)))).*sinh(sqrt((h(2,1).*P)./(k(1).*a)).*(L-x)))./(cosh(sqrt((h(2,1).*P)./(k(1).*a)).*L) +(h(2,1)./((sqrt((h(2,1).*P)./(k(1).*a))*k(1)))).*sinh(sqrt((h(2,1).*P)./(k(1).*a))*L))),'b')
hold on
errorbar(xdist,brass(2,:),0.5*ones(1,numel(xdist)), 'b.','HandleVisibility','off')

% plot(xdist,T_exact_fo(2,:),'r')
fplot(@(x) T_inf + (copper(2,1)- T_inf).*((cosh(sqrt((h(2,2).*P)./(k(2).*a))*(L-x)) + (h(2,2)./((sqrt((h(2,2).*P)./(k(2).*a)).*k(2)))).*sinh(sqrt((h(2,2).*P)./(k(2).*a)).*(L-x)))./(cosh(sqrt((h(2,2).*P)./(k(2).*a)).*L) +(h(1,2)./((sqrt((h(2,2).*P)./(k(2).*a))*k(2)))).*sinh(sqrt((h(2,2).*P)./(k(2).*a))*L))),'r')
errorbar(xdist,copper(2,:),0.5*ones(1,numel(xdist)), 'r.','HandleVisibility','off')

% plot(xdist,T_exact_fo(3,:),'m')
fplot(@(x) T_inf + (steel(2,1)- T_inf).*((cosh(sqrt((h(2,3).*P)./(k(3).*a)).*(L-x)) + (h(2,3)./((sqrt((h(2,3).*P)./(k(3).*a)).*k(3)))).*sinh(sqrt((h(2,3).*P)./(k(3).*a)).*(L-x)))./(cosh(sqrt((h(2,3).*P)./(k(3).*a)).*L) +(h(2,3)./((sqrt((h(2,3).*P)./(k(3).*a))*k(3)))).*sinh(sqrt((h(2,3).*P)./(k(3).*a))*L))),'m')
errorbar(xdist,steel(2,:),0.5*ones(1,numel(xdist)), 'm.','HandleVisibility','off')

% plot(xdist,T_exact_fo(4,:),'g')
fplot(@(x) T_inf + (alum(2,1)- T_inf).*((cosh(sqrt((h(2,4).*P)./(k(4).*a)).*(L-x)) + (h(2,4)./((sqrt((h(2,4).*P)./(k(4).*a)).*k(4)))).*sinh(sqrt((h(2,4).*P)./(k(4).*a)).*(L-x)))./(cosh(sqrt((h(2,4).*P)./(k(4).*a)).*L) +(h(2,4)./((sqrt((h(2,4).*P)./(k(4).*a))*k(4)))).*sinh(sqrt((h(2,4).*P)./(k(4).*a))*L))),'g')
errorbar(xdist,alum(2,:),0.5*ones(1,numel(xdist)), 'g.','HandleVisibility','off')

xlabel('Fin Length (m)')
ylabel('Temperature (Celsius)')
title('Temperature Distribution vs Fin Length (Forced Convection)')
legend('Brass','Copper','Steel','Aluminum')
xlim([0 L])
ylim([0 65])
hold off

%% Find q_fin values
q_fin = [];

% Free
q_fin(1,1) = sqrt(h(1,1).*P.*k(1)*a).*(brass(1,1)- T_inf).*((cosh(sqrt((h_br_fo.*P)./(k(1).*a)).*L) + (h_br_fo./((sqrt((h_br_fo.*P)./(k(1).*a)).*k(1)))).*sinh(sqrt((h_br_fo.*P)./(k(1).*a)).*L))./(cosh(sqrt((h_br_fo.*P)./(k(1).*a)).*L) +(h_br_fo./((sqrt((h_br_fo.*P)./(k(1).*a))*k(1)))).*sinh(sqrt((h_br_fo.*P)./(k(1).*a))*L)));
q_fin(1,2) = sqrt(h(1,2).*P.*k(2)*a).*(copper(1,1)- T_inf).*((cosh(sqrt((h_cu_fo.*P)./(k(2).*a)).*L) + (h_cu_fo./((sqrt((h_cu_fo.*P)./(k(2).*a)).*k(2)))).*sinh(sqrt((h_cu_fo.*P)./(k(2).*a)).*L))./(cosh(sqrt((h_cu_fo.*P)./(k(2).*a)).*L) +(h_cu_fo./((sqrt((h_cu_fo.*P)./(k(2).*a))*k(2)))).*sinh(sqrt((h_cu_fo.*P)./(k(2).*a))*L)));
q_fin(1,3) = sqrt(h(1,3).*P.*k(3)*a).*(steel(1,1)- T_inf).*((cosh(sqrt((h_st_fo.*P)./(k(3).*a)).*L) + (h_st_fo./((sqrt((h_st_fo.*P)./(k(3).*a)).*k(3)))).*sinh(sqrt((h_st_fo.*P)./(k(3).*a)).*L))./(cosh(sqrt((h_st_fo.*P)./(k(3).*a)).*L) +(h_st_fo./((sqrt((h_st_fo.*P)./(k(3).*a)).*k(3)))).*sinh(sqrt((h_st_fo.*P)./(k(3).*a)).*L)));
q_fin(1,4) = sqrt(h(1,4).*P.*k(4)*a).*(alum(1,1)- T_inf).*((cosh(sqrt((h_al_fo.*P)./(k(4).*a)).*L) + (h_al_fo./((sqrt((h_al_fo.*P)./(k(4).*a)).*k(4)))).*sinh(sqrt((h_al_fo.*P)./(k(4).*a)).*L))./(cosh(sqrt((h_al_fo.*P)./(k(4).*a)).*L) +(h_al_fo./((sqrt((h_al_fo.*P)./(k(4).*a)).*k(4)))).*sinh(sqrt((h_al_fo.*P)./(k(4).*a)).*L)));

% Forced
q_fin(2,1) = sqrt(h(2,1).*P.*k(1)*a).*(brass(2,1)- T_inf).*((cosh(sqrt((h_br_fo.*P)./(k(1).*a)).*L) + (h_br_fo./((sqrt((h_br_fo.*P)./(k(1).*a)).*k(1)))).*sinh(sqrt((h_br_fo.*P)./(k(1).*a)).*L))./(cosh(sqrt((h_br_fo.*P)./(k(1).*a)).*L) +(h_br_fo./((sqrt((h_br_fo.*P)./(k(1).*a))*k(1)))).*sinh(sqrt((h_br_fo.*P)./(k(1).*a))*L)));
q_fin(2,2) = sqrt(h(2,2).*P.*k(2)*a).*(copper(2,1)- T_inf).*((cosh(sqrt((h_cu_fo.*P)./(k(2).*a)).*L) + (h_cu_fo./((sqrt((h_cu_fo.*P)./(k(2).*a)).*k(2)))).*sinh(sqrt((h_cu_fo.*P)./(k(2).*a)).*L))./(cosh(sqrt((h_cu_fo.*P)./(k(2).*a)).*L) +(h_cu_fo./((sqrt((h_cu_fo.*P)./(k(2).*a))*k(2)))).*sinh(sqrt((h_cu_fo.*P)./(k(2).*a))*L)));
q_fin(2,3) = sqrt(h(2,3).*P.*k(3)*a).*(steel(2,1)- T_inf).*((cosh(sqrt((h_st_fo.*P)./(k(3).*a)).*L) + (h_st_fo./((sqrt((h_st_fo.*P)./(k(3).*a)).*k(3)))).*sinh(sqrt((h_st_fo.*P)./(k(3).*a)).*L))./(cosh(sqrt((h_st_fo.*P)./(k(3).*a)).*L) +(h_st_fo./((sqrt((h_st_fo.*P)./(k(3).*a)).*k(3)))).*sinh(sqrt((h_st_fo.*P)./(k(3).*a)).*L)));
q_fin(2,4) = sqrt(h(2,4).*P.*k(4)*a).*(alum(2,1)- T_inf).*((cosh(sqrt((h_al_fo.*P)./(k(4).*a)).*L) + (h_al_fo./((sqrt((h_al_fo.*P)./(k(4).*a)).*k(4)))).*sinh(sqrt((h_al_fo.*P)./(k(4).*a)).*L))./(cosh(sqrt((h_al_fo.*P)./(k(4).*a)).*L) +(h_al_fo./((sqrt((h_al_fo.*P)./(k(4).*a)).*k(4)))).*sinh(sqrt((h_al_fo.*P)./(k(4).*a)).*L)));
q_fin
