classdef BinaryLogisticModel < handle
    %LOGISTICMODEL Summary of this class goes here
    %   Detailed explanation goes here

    properties
        X   % (m × (n+1)) Eingabematrix inkl. Bias-Term
        y   % (m × 1) Zielvariable, nur 0 oder 1
        B   % ((n+1) × 1) Parametervektor (beta)
        H   % (m × 1) Hypothesen: sigmoid(X * B)
        mu     % Mittelwerte der Eingabevariablen (für Rückskalierung)
        sigma  % Standardabweichungen (für Rückskalierung)
        lambda
        regularization
    end

    methods
        function obj = BinaryLogisticModel(X, y)
            if nargin == 0
                % Default constructor: for preallocation
                obj.X = [];
                obj.y = [];
                obj.B = [];
                obj.H = [];
                obj.mu = [];
                obj.sigma = [];
                obj.lambda = 0;
                obj.regularization = "none";

            else
        
                % Bias hinzufügen
                m = size(X, 1);
                X_aug = [ones(m, 1), X];
            
                % Prüfen, ob binär
                uniqueY = unique(y);
                if ~all(ismember(uniqueY, [0 1]))
                    error('y muss binär sein: nur 0 und 1 sind erlaubt.');
                end
            
                % Initialisierung
                obj.X = X_aug;
                obj.y = y(:);  % Spaltenvektor
                obj.B = zeros(size(X_aug, 2), 1);
                obj.H = [];
                obj.mu = [];
                obj.sigma = [];
                obj.lambda = 0;
                obj.regularization = "none";
            end
        end
        
        function h = sigmoid(~, z)
            h = 1 ./ (1 + exp(-z));
        end

        function [C, gC] = computeCost(obj, beta)
            % Berechnet Kosten und Gradienten
            % Wenn beta nicht übergeben wird → verwende obj.B
        
            if nargin < 2
                beta = obj.B;
            end
        
            m = length(obj.y);
            z = obj.X * beta;
            h = obj.sigmoid(z);
        
            % numerisch stabilisieren (gegen log(0))
            h = min(max(h, 1e-15), 1 - 1e-15);
        
            obj.H = h;
            
            [costPenalty, gradPenalty] = obj.getPenalty();
            % Kostenfunktion (skalare Ausgabe)
            C = -mean(obj.y .* log(h) + (1 - obj.y) .* log(1 - h)) + costPenalty;
        
            % Gradienten (gleiche Dimension wie beta)
            gC = (obj.X' * (h - obj.y)) / m + gradPenalty;
        end
        
        function C = trainGradientDescent(obj, alpha, iter)
            % Trainiert das Modell mit Gradient Descent
            % alpha: Lernrate
            % iter: Anzahl Iterationen
            % Rückgabe: Verlauf der Kostenfunktion
        
            C = zeros(iter,1);
            for i = 1:iter
                [C0, gC] = obj.computeCost();
                C(i) = C0;
                obj.B = obj.B - alpha * gC;
            end
        end

        function C = trainFminunc(obj, maxIter,log)
            % Trainiert das Modell mit fminunc
            % maxIter: Maximale Iterationen
            % Rückgabe: finaler Kostenwert (Skalar)

            if nargin < 3
                log = false;
            end
        
            options = optimoptions('fminunc', ...
                'Algorithm', 'trust-region', ...
                'GradObj', 'on', ...
                'MaxIter', maxIter);
        
            initial_beta = zeros(size(obj.X, 2), 1);
            costFunc = @(b) obj.computeCost(b);  % closure mit obj
        
            [optB, Cval] = fminunc(costFunc, initial_beta, options);
        
            obj.B = optB;
            C = Cval;
            if log
                fprintf('Kostenfunktion beim Optimalwert: %f\n', C);
                fprintf('Optimales beta:\n');
                fprintf(' %f\n', obj.B);
            end
        end


        function scaleInputs(obj)
            X = obj.X;
            n = size(X, 2);
            obj.mu = zeros(1, n);
            obj.sigma = ones(1, n);  % Default-Werte für spätere Rückskalierung
    
            for j = 2:n  % Spalte 1 bleibt Bias
                mu_j = mean(X(:,j));
                sigma_j = std(X(:,j));
    
                X(:,j) = (X(:,j) - mu_j) / sigma_j;
    
                obj.mu(j) = mu_j;
                obj.sigma(j) = sigma_j;
            end
    
            obj.X = X;
        end

        function B_rescaled = rescaleParameters(obj)
            % Rückskalierung gemäß:
            % β₀ = ~β₀ - sum(~βᵢ * μᵢ / σᵢ)
            % βᵢ = ~βᵢ / σᵢ
    
            B_scaled = obj.B;
            mu = obj.mu;
            sigma = obj.sigma;
            n = length(B_scaled);
    
            B_rescaled = zeros(n, 1);
    
            % Rückskalierte Steigungen: βᵢ = ~βᵢ / σᵢ  (i ≥ 1)
            for i = 2:n
                B_rescaled(i) = B_scaled(i) / sigma(i);
            end
    
            % Rückskalierter Achsenabschnitt:
            % β₀ = ~β₀ - ∑ ~βᵢ * μᵢ / σᵢ
            correction = 0;
            for i = 2:n
                correction = correction + B_scaled(i) * mu(i) / sigma(i);
            end
            B_rescaled(1) = B_scaled(1) - correction;
            obj.B = B_rescaled();
        end
        
        function ER = accuracy(obj, Xval, yval)
            % Berechnet die Erkennungsrate für gegebene Validierungsdaten
            h = obj.sigmoid(Xval * obj.B);
            y_hat = h >= 0.5;
            ER = mean(y_hat == yval);
        end

        function models = createPolyModel(obj, degrees)
            % Erstellt ein Array von BinaryLogisticModel-Objekten für gegebene Polynomialgrade
            % degrees: Vektor von Polynomialgraden (z.B. [2, 3, 5])
            % Rückgabe: Array von BinaryLogisticModel-Instanzen
        
            if size(obj.X, 2) ~= 3
                error('createPolyModel funktioniert nur mit genau 2 Features (plus Bias-Spalte).');
            end
        
            x1 = obj.X(:,2);
            x2 = obj.X(:,3);
        
            numDegrees = length(degrees);
            models(1, numDegrees) = regression.BinaryLogisticModel();  % Preallocate model array
        
            for k = 1:numDegrees
                degree = degrees(k);
        
                % Polynommerkmale erzeugen
                features = ones(size(x1));  % Start mit Bias-Spalte
                for i = 1:degree
                    for j = 0:i
                        features(:, end + 1) = (x1.^(i - j)) .* (x2.^j);
                    end
                end
        
                % Modell erzeugen (Bias-Spalte wird nicht nochmal hinzugefügt)
                models(k) = regression.BinaryLogisticModel(features(:, 2:end), obj.y);
            end
        end

        function [R, lambda] = eigenvalues(obj)
            % Gibt die Korrelationsmatrix R und ihre Eigenwerte lambda zurück
            m = size(obj.X, 1);
            R = (1 / m) * (obj.X' * obj.X);  % Korrelationsmatrix
            lambda = eig(R);                 % Eigenwerte
        end

        function [costPenalty, gradPenalty] = getPenalty(obj)
            % Berechnet die Regularisierungsanteile für Kostenfunktion und Gradienten
            % Nur für L2 (ridge) Regularisierung
            
            if isempty(obj.B)
                error("getPenalty: Parameter 'B' ist leer.");
            end
        
            if isempty(obj.X)
                error("getPenalty: Trainingsdaten 'X' sind leer, m kann nicht berechnet werden.");
            end
        
            m = size(obj.X, 1);  % Anzahl Trainingsbeispiele
            B = obj.B;
        
            switch obj.regularization
                case "ridge"
                    % Ignoriere Bias-Term (B(1))
                    BReg = [0; B(2:end)];
        
                    costPenalty = (obj.lambda / (2 * m)) * sum(BReg.^2);
                    gradPenalty = (obj.lambda / m) * BReg;
        
                case "lasso"
                    warning("Lasso: Gradienten-Term ist nicht exakt definiert (nicht differenzierbar).");
                    BReg = [0; B(2:end)];
        
                    costPenalty = (obj.lambda / m) * sum(abs(BReg));
                    gradPenalty = (obj.lambda / m) * sign(BReg);  % Platzhalter – siehe Hinweis
        
                case "none"
                    costPenalty = 0;
                    gradPenalty = zeros(size(obj.B));
        
                otherwise
                    error("Unbekannter Regularisierungstyp '%s'", obj.regularization);
            end
        end


    end
end