classdef LogisticModel < handle
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
        reg
        X_current   
    end

    methods
        function obj = LogisticModel(X, y)
            if nargin == 0
                % Default constructor: for preallocation
                obj.X = [];
                obj.y = [];
                obj.B = [];
                obj.H = [];
                obj.mu = [];
                obj.sigma = [];
                obj.lambda = 0;
                obj.reg = "L2";

            else
        
                % Bias hinzufügen
                m = size(X, 1);
            
                % Prüfen, ob binär
                uniqueY = unique(y);
                if ~all(ismember(uniqueY, [0 1]))
                    error('y muss binär sein: nur 0 und 1 sind erlaubt.');
                end
            
                % Initialisierung
                obj.X = X;
                obj.X_current = [ones(m, 1), X];
                obj.y = y(:);  % Spaltenvektor
                obj.B = zeros(size(obj.X_current, 2), 1);
                obj.H = [];
                obj.mu = [];
                obj.sigma = [];
                obj.lambda = 0;
                obj.reg = "L2";
            end
        end
        
        function h = sigmoid(~, z)
            h = 1 ./ (1 + exp(-z));
        end

        function [C, gC] = cost(obj, beta)
            % Berechnet Kosten und Gradienten
            % Wenn beta nicht übergeben wird → verwende obj.B
            if nargin > 1
                obj.B = beta;
            end
        
            m = length(obj.y);
            z = obj.X_current * obj.B;
            h = obj.sigmoid(z);
        
            % numerisch stabilisieren (gegen log(0))
            h = min(max(h, 1e-15), 1 - 1e-15);
        
            obj.H = h;
            
            % Kostenfunktion (skalare Ausgabe)
            C = -mean(obj.y .* log(h) + (1 - obj.y) .* log(1 - h)) + obj.getCostPenalty();
        
            % Gradienten (gleiche Dimension wie beta)
            gC = (obj.X_current' * (h - obj.y)) / m + obj.getGradPenalty();
        end
        
        function C = trainGradientDescent(obj, alpha, iter)
            % Trainiert das Modell mit Gradient Descent
            % alpha: Lernrate
            % iter: Anzahl Iterationen
            % Rückgabe: Verlauf der Kostenfunktion
            obj.scaleInputs();
            C = zeros(iter,1);
            for i = 1:iter
                [C0, gC] = obj.cost();
                C(i) = C0;
                obj.B = obj.B - alpha * gC;
            end
            obj.B = obj.rescaleParameters();

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
            initial_beta = zeros(size(obj.X_current, 2), 1);
            costFunc = @(b) obj.cost(b);  % closure mit obj
        
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
            m = size(obj.X,1);
            obj.X_current = [ones(m,1),obj.X]

            X = obj.X_current;
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
    
            obj.X_current = X;
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
            obj.B = B_rescaled;
            obj.resetCurrentX()
        end
        
        function [ER, recall, precision, CM] = accuracy(obj, Xval, yval)
            % Berechnet Accuracy, Recall und Precision für Klassifikation
            
            if isempty(obj.B)
                error('Das Modell muss zuerst mit fit() trainiert werden.');
            end
        
            % Design-Matrix mit Bias
            CM = obj.confusionMatrix(Xval,yval);
            Xval = [yval.^0, Xval];
            % Vorhersage mit Schwellenwert 0.5
            h = obj.sigmoid(Xval * obj.B);
            y_hat = h >= 0.5;
        
            % Metriken berechnen
            TP = sum((yval == 1) & (y_hat == 1));  % True Positives
            FP = sum((yval == 0) & (y_hat == 1));  % False Positives
            FN = sum((yval == 1) & (y_hat == 0));  % False Negatives
        
            ER = mean(y_hat == yval);  % Accuracy
            recall = TP / (TP + FN);   % Sensitivität
            precision = TP / (TP + FP);% Positiver Vorhersagewert
            
        end


        function models = createPolyModel(obj, degrees)
            % Erstellt ein Array von LogisticModel-Objekten für gegebene Polynomialgrade
            % degrees: Vektor von Polynomialgraden (z.B. [2, 3, 5])
            % Rückgabe: Array von LogisticModel-Instanzen
        
            if size(obj.X, 2) ~= 3
                error('createPolyModel funktioniert nur mit genau 2 Features (plus Bias-Spalte).');
            end
        
            x1 = obj.X(:,2);
            x2 = obj.X(:,3);
        
            numDegrees = length(degrees);
            models(1, numDegrees) = regression.LogisticModel();  % Preallocate model array
        
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
                models(k) = regression.LogisticModel(features(:, 2:end), obj.y);
            end
        end

        function [R, lambda] = eigenvalues(obj)
            % Gibt die Korrelationsmatrix R und ihre Eigenwerte lambda zurück
            m = size(obj.X, 1);
            R = (1 / m) * (obj.X_current' * obj.X_current);  % Korrelationsmatrix
            lambda = eig(R);                 % Eigenwerte
        end

        function penalty = getCostPenalty(obj)
            if isempty(obj.lambda) || obj.lambda == 0
                penalty = 0;
                return;
            end
        
            m = size(obj.X, 1);
        
            switch upper(obj.reg)
                case 'L2'
                    penalty = (obj.lambda / (2 * m)) * sum(obj.B(2:end).^2);  % ohne Bias
                case 'L1'
                    penalty = (obj.lambda / m) * sum(abs(obj.B(2:end)));     % ohne Bias
                otherwise
                    error('Unbekannter Regularisierungstyp: %s', obj.reg);
            end
        end
        
        function penalty = getGradPenalty(obj)
            if isempty(obj.lambda) || obj.lambda == 0
                penalty = zeros(size(obj.B));
                return;
            end
        
            m = size(obj.X, 1);
        
            switch upper(obj.reg)
                case 'L2'
                    penalty = (obj.lambda / m) * [0; obj.B(2:end)];
                case 'L1'
                    penalty = (obj.lambda / m) * sign([0; obj.B(2:end)]);
                otherwise
                    error('Unbekannter Regularisierungstyp: %s', obj.reg);
            end
        end

        function new_model = createPolyFeatures(obj, index, powers)
            % Erzeugt ein neues Modell mit Polynommen für eine gegebene Spalte
            % index: Spaltenindex der originalen Eingabe (z. B. 1 oder 2)
            % powers: Vektor mit Exponenten (z. B. [1 2 3])
        
            x = obj.X(:, index);        % Original (nicht X_current!)
            n = size(x, 1);
            num_terms = length(powers);
            X_poly = zeros(n, num_terms);
            for j = 1:num_terms
                X_poly(:, j) = x.^powers(j);
            end
        
            % Neues Modell mit diesen Features
            new_model = regression.LogisticModel(X_poly, obj.y);
        end

        function resetCurrentX(obj)
            obj.X_current = [ones(size(obj.X, 1), 1), obj.X];
        end

        function CM = confusionMatrix(obj, Xval, yval)
            % Berechnet die Konfusionsmatrix für gegebene Validierungsdaten
            % Rückgabeformat: 2x2 Matrix [TP FN; FP TN]
            
            if isempty(obj.B)
                error('Das Modell muss zuerst trainiert werden.');
            end

            % Design-Matrix mit Bias
            Xval = [ones(size(Xval, 1), 1), Xval];

            % Vorhersage
            h = obj.sigmoid(Xval * obj.B);
            y_hat = h >= 0.5;

            % Elemente der Konfusionsmatrix
            TP = sum((yval == 1) & (y_hat == 1));
            FN = sum((yval == 1) & (y_hat == 0));
            FP = sum((yval == 0) & (y_hat == 1));
            TN = sum((yval == 0) & (y_hat == 0));

            % Konfusionsmatrix
            CM = [TP, FN; FP, TN];
        end


    end
end