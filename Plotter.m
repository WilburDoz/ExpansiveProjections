%% Plotter
% Takes in the standard inputs of running sim and plots them in standard
% way.
%
% INPUTS:
% Data - Dim x N set of datapoints
% Labels - 1 x N set of labels
% Assignments_Basic - best linear cut through data assignments
% Dimensions - list of second layer dimensions tried
% Accuracy_Proj_Mean - 1 x Dimensions average accuracy for each dimension
% Accuracy_Proj_Dev - 1x Dimensions std dev of accuracies for each dim
% Accuracy_Basic - Linear perceptron accuracy, for comparison
% Stored_Assignment - best case assignments
% J - random projection weight matrix
%
% Plots 5 graphs about these facts.
%%
function [] = Plotter(Data, Labels, Assignments_Basic, Dimensions, ...
    Accuracy_Proj_Mean, Accuracy_Proj_Dev, Accuracy_Basic, stored_assignments,...
    J)
Dim = size(Data, 1);

figure
subplot(2,3,1)
if Dim == 2
    hold on
    plot(Data(1,Labels == 1), Data(2,Labels == 1), 'r*')
    plot(Data(1,Labels == -1), Data(2,Labels == -1), 'b*')
elseif Dim == 3
    plot3(Data(1,Labels == 1), Data(2,Labels == 1), Data(3,Labels==1), 'r*')
    hold on
    plot3(Data(1,Labels == -1), Data(2,Labels == -1), Data(3,Labels == -1),'b*')    
end
title('Data Distribution')

subplot(2,3,2)
if Dim == 2
    hold on
    plot(Data(1,Assignments_Basic == 1), Data(2,Assignments_Basic == 1), 'r*')
    plot(Data(1,Assignments_Basic == -1), Data(2,Assignments_Basic == -1), 'b*')
elseif Dim == 3
    plot3(Data(1,Assignments_Basic == 1), Data(2,Assignments_Basic == 1), Data(3,Assignments_Basic==1), 'r*')
    hold on
    plot3(Data(1,Assignments_Basic == -1), Data(2,Assignments_Basic == -1), Data(3,Assignments_Basic == -1),'b*')    
end
title('Best Performance with Linear Readout')

subplot(2,3,3)
hold on
errorbar(Dimensions, Accuracy_Proj_Mean, Accuracy_Proj_Dev,'DisplayName','Random Projections')
title('Accuracy vs Dimensionality')
xlabel('Dimension')
ylabel('Accuracy')
plot([min(Dimensions), max(Dimensions)], [Accuracy_Basic, Accuracy_Basic], 'DisplayName', 'Linear Decoder')
plot([Dim*50, Dim*50], [0.5,1], 'DisplayName', 'Fly-like 50 fold expansion')
legend

subplot(2,3,4)
if Dim == 2
    hold on
    plot(Data(1,stored_assignments == 1), Data(2,stored_assignments == 1), 'r*')
    plot(Data(1,stored_assignments == -1), Data(2,stored_assignments == -1), 'b*')
elseif Dim == 3
    plot3(Data(1,stored_assignments == 1), Data(2,stored_assignments == 1), Data(3,stored_assignments==1), 'r*')
    hold on
    plot3(Data(1,stored_assignments == -1), Data(2,stored_assignments == -1), Data(3,stored_assignments == -1),'b*')    
end
title('Best Performing Random')

subplot(2,3,5)
if Dim == 2
    plot(J(:,1),J(:,2),'.')
elseif Dim == 3
    plot3(J(:,1), J(:,2), J(:,3),'.')
end
hold on
title('Placement of Random Weights')
hold off