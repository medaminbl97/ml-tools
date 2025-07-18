classdef svmModel < handle
    % svmModel - Klasse für Support Vector Machines mit fitcsvm.
    % Beispiel:
    % model = svmModel(X, y);
    % model.kernelFunction = 'rbf';
    % model.boxConstraint = 1;
    % model.train();
    % labels = model.predict(Xtest);
    % model.plotDecisionBoundary();
    % model.showModelInfo();

    properties
        X % Trainingsdaten
        y % Labels
        kernelFunction = 'linear' % Kernel-Funktion ('linear', 'rbf', etc.)
        boxConstraint = 1         % Box-Constraint (C)
        Model % Enthält das trainierte ClassificationSVM-Objekt
    end

    methods
        function obj = svmModel(X, y)
            % Konstruktor: Speichert X und y
            if nargin > 0
                obj.X = X;
                obj.y = y;
            end
        end

        function train(obj,varargin)
            % Trainiert das SVM-Modell mit den gespeicherten Parametern
        
            % Basis-Argumente
            args = {
                'KernelFunction', obj.kernelFunction, ...
                'BoxConstraint', obj.boxConstraint, ...
                'Standardize', true
            };

                        % Zusätzliche Name-Value-Pairs hinzufügen
            if ~isempty(varargin)
                args = [args, varargin];
            end
        
            % Aufruf
            obj.Model = fitcsvm(obj.X, obj.y, args{:});
        end

        function labels = predict(obj, Xtest)
            % Gibt Vorhersagen für neue Daten zurück
            labels = predict(obj.Model, Xtest);
        end

        function sv = getSupportVectors(obj)
            % Gibt die Support Vektoren zurück
            sv = obj.Model.SupportVectors;
        end

        function [bestC,bestScale] = optimizeHyperparameters(obj, CexpArray, scaleExpArray)
            % Standardwerte, falls nichts übergeben wird
            if nargin < 2 || isempty(CexpArray)
                CexpArray = [-2, -1, 0, 1, 2];  % entspricht [0.01, ..., 100]
            end
            if nargin < 3 || isempty(scaleExpArray)
                scaleExpArray = [-2, -1, 0, 1, 2];
            end
        
            Cvals = 10.^CexpArray;
            scaleVals = 10.^scaleExpArray;
        
            bestC = NaN;
            bestScale = NaN;
            bestLoss = inf;
        
            % Grid Search über beide Parameter
            for C = Cvals
                for s = scaleVals
                    % Cross-Validated SVM trainieren
                    Mdl = fitcsvm(obj.X, obj.y, ...
                        'KernelFunction', obj.kernelFunction, ...
                        'BoxConstraint', C, ...
                        'KernelScale', s, ...
                        'Standardize', true, ...
                        'CrossVal', 'on');
        
                    loss = kfoldLoss(Mdl);
        
                    if loss < bestLoss
                        bestLoss = loss;
                        bestC = C;
                        bestScale = s;
                    end
                end
            end
        
            fprintf('✅ Optimale Hyperparameter:\n');
            fprintf('BoxConstraint (C): %.4f\n', bestC);
            fprintf('KernelScale: %.4f\n', bestScale);
            fprintf('CV-Verlust: %.4f\n', bestLoss);
        
            obj.train();
        end



        function showModelInfo(obj)
            % Zeigt Informationen über das Modell an
            fprintf('Bias (Bias-Term): %.4f\n', obj.Model.Bias);
            fprintf('Anzahl Support Vektoren: %d\n', size(obj.Model.SupportVectors, 1));
            if isprop(obj.Model, 'Beta') && ~isempty(obj.Model.Beta)
                disp('Beta (Gewichte):');
                disp(obj.Model.Beta);
            else
                disp('Beta ist für diesen Kernel nicht definiert (z. B. nicht-linearer Kernel).');
            end
        end
        function plotBoundary(obj,varargin)
            %plotDecisionBoundary(X,SVMModel,x1min, x1max, x2min, x2max, N)
            %Zeichnet die Entscheidungsschwelle
            %alle Parameter ab x1min sind optional
            
            Largin = length(varargin);
            %Erzeugen eines meshgrids
            if Largin>=5
                N = varargin{5};
            else
            N=200;
            end
            if Largin>=4
                x1min = varargin{1};
                x1max = varargin{2};
                x2min = varargin{3};
                x2max = varargin{4};
                x1 = linspace(x1min, x1max, N);
                x2 = linspace(x2min, x2max, N);        
            else
                x1=linspace(min(obj.X(:,1)),max(obj.X(:,1)),N);
                x2=linspace(min(obj.X(:,2)),max(obj.X(:,2)),N);
            end
            [X1,X2] = meshgrid(x1, x2);
            %Erzeugen des scoreGrids
            [~,score] = predict(obj.Model,[X1(:),X2(:)]);
            scoreGrid = reshape(score(:,1),size(X1,1),size(X2,2));
            
            %Zeichnen der Entscheidungsschwelle
            contour(X1,X2,scoreGrid,[0, 0],'LineColor','k','Linewidth',1.6)
            
            %Zeichnen des Margins
            contour(X1,X2,scoreGrid,[1, 1],'k:','Linewidth',0.5)
            contour(X1,X2,scoreGrid,[-1, -1],'k:','Linewidth',0.5)
        end
        function plotsv(obj)
            %Markiert die Support Vektoren
            
            %Indizes der Support Vektoren
            svInd = obj.Model.IsSupportVector;
            %Markieren der Support Vektoren
            plot(obj.X(svInd,1),obj.X(svInd,2),'ko','MarkerSize',10)
        end

        function [ER,FR] = accuracy(obj, Xtest, ytest)
            % Evaluates the model on test data and returns the error rate (Fehlerrate)
            %
            % FR = 1 - (correctly classified samples / total samples)
        
            % Vorhersage
            y_pred = predict(obj.Model, Xtest);
        
            % Erkennungsrate
            ER = sum(y_pred == ytest) / numel(ytest);
        
            % Fehlerrate
            FR = 1 - ER;
        end


    end
end
