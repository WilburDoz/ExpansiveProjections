%% 4th April - Composing Shapes III
% Choose parameters
Lambda = 0.0000001;
N = 5000;
Dim = 2;
Option = 2;
p1 = 40;
Weight_Option = 1;
Sparseness = 0.5;
Num = 1;
Dimensions = [1,2,3,10,50,100,200,400,600,1000];%,2000];
Thresh_Option = 2;

% Setup data
[Data, Labels, N]  = CreateDistribution(N, Option, Dim, p1,0,0);

% Now train a linear classifier on this, how well does it do?
Data2 = [Data; ones(1,N)];
Weights_Basic = PerceptronWeights(Data2', Labels', 2, Lambda);
Assignments_Basic = sign(Weights_Basic'*Data2);
Accuracy_Basic = sum(Assignments_Basic == Labels)/N;

% Now we want to do random projections into various dimensionalities and measure the accuracy
Accuracy_Proj = zeros(length(Dimensions), Num);
maxacc = 0;
for k = 1:Num
    disp(['Num = ', num2str(k)])
    for j = 1:length(Dimensions)
        Dim_Proj = Dimensions(j);
        %disp(['Dimension = ',num2str(Dim)])
        if Weight_Option == 0
            J = randn([Dim_Proj, Dim]);
            for i = 1:Dim_Proj
                J(i,:) = J(i,:)/norm(J(i,:));
            end
        elseif Weight_Option == 1
            J = 2*(randn([Dim_Proj,Dim]) - repmat(0.5,[Dim_Proj,Dim]));
        end
        
        ProjData = [Threshold(J*Data,0,Sparseness,Thresh_Option); ones(1,N)];
        
        Weights_Proj = PerceptronWeights(ProjData',Labels', 2, Lambda);
        Assignments = sign(Weights_Proj'*ProjData);
        Accuracy_Proj(j,k) = sum(Assignments == Labels)/N;
        if Accuracy_Proj(j,k) > maxacc
            maxacc = Accuracy_Proj(j,k);
            stored_assignments = Assignments;
            Weights_Store = Weights_Proj;
        end            
    end
end
Accuracy_Proj_Mean = mean(Accuracy_Proj, 2);
Accuracy_Proj_Dev = std(Accuracy_Proj, 0, 2);

% And plot
Plotter(Data, Labels, Assignments_Basic, Dimensions, ...
    Accuracy_Proj_Mean, Accuracy_Proj_Dev, Accuracy_Basic,... 
    stored_assignments, J)