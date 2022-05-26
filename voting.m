function [predicted_final,votes]=voting(predicted_original,predicted_deriv1,predicted_deriv2)

% predicted_original : labels obtained by GEPSVM hyperplane using original data
% predicted_deriv1 : labels obtained by GEPSVM hyperplane using 1st derivative data
% predicted_deriv2 : labels obtained by GEPSVM hyperplane using 2nd derivative data
% votes : contains votes of each class column wise
% predicted_final : final labels obtained by voting scheme

class1=1;class2=-1;
predicted_final=[];
votes=[];
votes=zeros(size(predicted_original,1),2);

for x=1:size(predicted_original,1)
    if (predicted_original(x,1)==class1)
        votes(x,1)=votes(x,1)+1;
    else 
        votes(x,2)=votes(x,2)+1;
    end
    if (predicted_deriv1(x,1)==class1)
        votes(x,1)=votes(x,1)+1;
    else 
        votes(x,2)=votes(x,2)+1;
    end
    if (predicted_deriv2(x,1)==class1)
        votes(x,1)=votes(x,1)+1;
    else 
        votes(x,2)=votes(x,2)+1;
    end
end

for x=1:size(votes,1)
    if(votes(x,1)>votes(x,2))
        predicted_final(x,1)=class1;
    else
        predicted_final(x,1)=class2;
    end
end
