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
    end

    methods
        function obj = BinaryLogisticModel(X, y)
            if nargin < 2
                error('Bitte X und y beim Erstellen des Modells angeben.');
            end
        
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
        end
        
        function h = sigmoid(~, z)
            h = 1 ./ (1 + exp(-z));
        end

        function [C, gC] = computeCost(obj)
            % Berechnet die Kostenfunktion und den Gradienten für das Modell
            m = length(obj.y);
            z = obj.X * obj.B;
            h = obj.sigmoid(z);  % benutzt interne sigmoid-Methode
            obj.H = h;
    
            % Kosten
            C = -(log(h') * obj.y + log(1 - h') * (1 - obj.y)) / m;
    
            % Gradienten
            gC = (obj.X' * (h - obj.y)) / m;
        end

        function C = train(obj, alpha, iter)
            % Führt Gradient Descent durch und aktualisiert obj.B
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

        function newModel = createPolyModel(obj, degree)
            % Wandelt X(:,2:3) in Polynommerkmale um (x1, x2)
            % Gibt ein neues BinaryLogisticModel mit erweiterten Features zurück
    
            if size(obj.X, 2) ~= 3
                error('createPolyModel funktioniert nur mit genau 2 Features (plus Bias-Spalte).');
            end
    
            x1 = obj.X(:,2);
            x2 = obj.X(:,3);
    
            % Polynommerkmale erzeugen
            out = zeros(size(x1));  % Bias-Spalte
            for i = 1:degree
                for j = 0:i
                    out(:, end + 1) = (x1.^(i-j)) .* (x2.^j);
                end
            end
    
            % Neues Modell erzeugen (ohne nochmal Bias-Spalte hinzufügen)
            newModel = regression.BinaryLogisticModel(out(:,2:end), obj.y);
        end

        function [R, lambda] = eigenvalues(obj)
            % Gibt die Korrelationsmatrix R und ihre Eigenwerte lambda zurück
            m = size(obj.X, 1);
            R = (1 / m) * (obj.X' * obj.X);  % Korrelationsmatrix
            lambda = eig(R);                 % Eigenwerte
        end

    end
end