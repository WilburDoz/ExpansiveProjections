%% CreateDistribution
% This function creates the initial layout of points for classification.
% Currently points are in the square between -half and half in 2D and have
% only two labels, +1 and -1.
%
% TODO: 1) Allow higher dimension
%           i) Nearly done, just need to think about neighbours in 3D
%       2) Allow multiclass
%       3) Allow range or shape specification?
%       5) The alignment ones (Option == 1) have this diagonal bias from
%       initialisation and method. Can we remove it?
%       6) Other methods? Should be normativly motivated. Are we intrested
%       in convolutedness of decision boundaries? Or does class number work
%       well enough?
%
% INPUTS:
% N - number of points
% Option - chooses which the scheme for point creation
% p1, p2, p3 - Option specific parameters one, two, and three
%
% Option 0: three concentric circles of alternating +1/-1 Label
%   p1: inner circle radius
%   p2: 2nd circle radius - if 0 then only one circle
%   p3: third circle radius - if 0 then only two circles
%
% Option 1: grid of points with a bias towards aligning the lables of
% neighbouring points to one and other.
%   p1: probability that you will align with your neighbour
%   p2: if 0 only 4 neighbours (side-to-side and up/down) affect behaviour
%       if 1 diagonally adjacent neighbours influence just as much
%
% Option 2: Class centroids with random labels are defined. Poitns are
% given the label of the nearest class
%   p1: M the number of classes
%
% OUTPUTS:
% Data - 2xN co-ordinates of points
% Labels - 1xN list of labels for points
% N - number of datapoint, since sometimes we round N conveniently, only
%       different to input in Option 1
%%
function [Data, Labels, N] = CreateDistribution(N, Option, Dim, p1, p2, p3)

% Check you have chosen a good option
Potential_Options = [0,1,2];
Good_Option = 0;
for i = 1:length(Potential_Options)
    if Option == Potential_Options(i)
        Good_Option = 1;
    end
end
if ~Good_Option
    error(['Option not correctly chosen, choose from: ', num2str(Potential_Options)])
end
if round(Dim) ~= Dim
    error('Dimension must be an integer')
end

if Option == 0
    % Rearrange parameters so they are in increasing order
    parameters = [p1,p2,p3];
    circles = sum(parameters ~= 0);
    radii = sort(parameters(parameters ~=0));
        
    % Create data
    Data = rand([Dim,N]) - repmat(0.5,Dim,N);
    Labels = ones([1,N]);

    % Assign labels
    for i = 1:N
        if circles > 0 && norm(Data(:,i)) < radii(1)
            Labels(i) = 1;
        elseif circles > 1 && norm(Data(:,i)) < radii(2)
            Labels(i) = -1;
        elseif circles > 2 && norm(Data(:,i)) < radii(3)
            Labels(i) = 1;
        else
            Labels(i) = -1*(-1)^(1+ circles);
        end
    end
elseif Option == 1
    if Dim ~= 2 && Dim ~= 3
        error('This method only works in 2 and 3 Dimensions')
    end
    if Dim == 3 && p2 == 1
        error('This combinations of dimension and neighbours has not been coded yet')
    end
    % Create grid of data
    Dim_One_N = round(N^(1/Dim));
    N = Dim_One_N^Dim;
    if Dim == 2
        [Data_X, Data_Y] = meshgrid(-0.5:1/(Dim_One_N-1):0.5, -0.5:1/(Dim_One_N-1):0.5);
        Data = [reshape(Data_X,[1,N]); reshape(Data_Y, [1,N])];
        Labels = zeros([Dim_One_N, Dim_One_N]);
        Labels(1,1) = 1; % Lets kick us off
    elseif Dim == 3
        [Data_X, Data_Y, Data_Z] = meshgrid(-0.5:1/(Dim_One_N-1):0.5, -0.5:1/(Dim_One_N-1):0.5, -0.5:1/(Dim_One_N-1):0.5);
        Data = [reshape(Data_X,[1,N]); reshape(Data_Y, [1,N]); reshape(Data_Z, [1,N])];        
        Labels = zeros([Dim_One_N, Dim_One_N, Dim_One_N]);
        Labels(1,1,1) = 1; % Lets kick us off
    end

    for i = 1:Dim_One_N
        for j = 1:Dim_One_N
            if Dim == 2
                if p2 == 0
                    p = p1; % prob you align with neighbour
                    % Then go through all the pixels below and left
                    Neighbours = sum(Labels(max(i-1,1),j) + Labels(i,max(1,j-1)));
                    if rand(1) < p
                        Labels(i,j) = sign(Neighbours);
                    else
                        Labels(i,j) = -sign(Neighbours);
                    end
                    if Labels(i,j) == 0
                        Labels(i,j) = sign(rand(1) - 0.5);
                    end
                elseif p2 == 1
                    % Then go through neighbours as well
                    p = 1 - p1; % effective antialignment effect of neighbour
                    Neighbours = sum(Labels(max(i-1,1),j) + Labels(i,max(1,j-1)) + Labels(max(i-1,1),max(j-1,1)) + Labels(max(i-1,1),min(j+1,Dim_One_N)));
                    Labels(i,j) = sign(Neighbours)*sign(rand(1) - (1-p)*p^(abs(Neighbours)-1));
                    if Labels(i,j) == 0
                        Labels(i,j) = sign(rand(1) - 0.5);
                    end
                end
            elseif Dim == 3
                for k = 1:Dim_One_N
                    if p2 == 0
                        p = p1; % prob you align with neighbour
                        % Then go through all the pixels below and left
                        Neighbours = sum(Labels(max(i-1,1),j,k) + Labels(i,max(1,j-1),k) + Labels(i,j,max(1,k-1)));
                        if rand(1) < p
                            Labels(i,j,k) = sign(Neighbours);
                        else
                            Labels(i,j,k) = -sign(Neighbours);
                        end
                        if Labels(i,j,k) == 0
                            Labels(i,j,k) = sign(rand(1) - 0.5);
                        end
                    elseif p2 == 1
                        % Then go through neighbours as well
                        p = 1 - p1; % effective antialignment effect of neighbour
                        Neighbours = sum(Labels(max(i-1,1),j,k) + Labels(i,max(1,j-1),k) + Labels(max(i-1,1),max(j-1,1),k) + Labels(max(i-1,1),min(j+1,Dim_One_N),k)...
                            + Labels(i,j,max(1,k-1)));
                        Labels(i,j) = sign(Neighbours)*sign(rand(1) - (1-p)*p^(abs(Neighbours)-1));
                        if Labels(i,j) == 0
                            Labels(i,j) = sign(rand(1) - 0.5);
                        end
                    end
                end
            end
        end
    end
    Labels = reshape(Labels,1,[]);
elseif Option == 2
    if round(p1) ~= p1
        error('This parameter is number of classes, must be integer')
    end
    Centroids = rand([Dim,p1]) - repmat(0.5, [Dim,p1]);
    Labels_Centroids = sign(rand([1,p1])-repmat(0.5,[1,p1]));
    Data = rand([Dim,N]) - repmat(0.5,Dim,N);
    Idx = knnsearch(Centroids', Data');
    Labels = Labels_Centroids(Idx);
end