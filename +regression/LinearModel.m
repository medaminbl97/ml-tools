classdef LinearModel < handle
    properties
        X           % Original Trainingsdaten (m x n)
        y           % Zielwerte (m x 1)
        B           % Parametervektor
        h           % Vorhersagen auf Trainingsdaten
        mu          % Spaltenmittelwerte (außer Bias)
        sigma       % Standardabweichungen (außer Bias)
        X_current   % Aktuelle Featureauswahl
        featureInds
    end

    methods
        function obj = LinearModel(X, y)
            obj.X = X;
            obj.y = y;
            obj.B = [];
            obj.h = [];
            obj.mu = [];
            obj.sigma = [];
            obj.X_current = X;
            obj.featureInds = [];
            obj.setFeatureInds();
        end

        function setFeatureInds(obj, Inds)
            if nargin < 2 || isempty(Inds)
                obj.X_current = obj.X;
                obj.featureInds = 1:size(obj.X, 2);
            else
                obj.X_current = obj.X(:, Inds);
                obj.featureInds = Inds;
            end
        end

        function fit(obj)
            X_design = [ones(size(obj.X_current, 1), 1), obj.X_current];
            obj.B = pinv(X_design) * obj.y;
            obj.h = X_design * obj.B;
        end

        function C = cost(obj)
            m = size(obj.X_current, 1);
            X_design = [ones(m, 1), obj.X_current];
            e = obj.y - X_design * obj.B;
            obj.h = X_design * obj.B;
            C = (1 / (2 * m)) * (e' * e);
        end

        function rmse = computeRMSE(obj,Xv,yv)
            % y_true: echte Werte
            % y_pred: Vorhersagewerte
            hv = obj.predict(Xv);
            rmse = sqrt(mean((yv - hv).^2));
        end

        function h = predict(obj, X_test)
            if isempty(obj.B)
                error('Model must be fitted before prediction.');
            end
            X_test = [ones(size(X_test, 1), 1), X_test(:,obj.featureInds)];
            h = X_test * obj.B;
        end

        function score = r2(obj, X_val, y_val, y_est)
            if nargin < 4 || isempty(y_est)
                if isempty(obj.B)
                    error('Model must be fitted before calling r2().');
                end
                X_val = [y_val.^0, X_val(:, obj.featureInds)];  % Add bias
                y_est = X_val * obj.B;
            end
        
            ss_res = sum((y_val - y_est).^2);
            ss_tot = sum((y_val - mean(y_val)).^2);
            score = 1 - (ss_res / ss_tot);
        end


        function [R, lambda] = eigenvalues(obj)
            % PCA auf X_current, ohne Bias und mit Standardisierung
        
            X = obj.X_current;
        
            % 1. Zentrieren und Standardisieren
            X_norm = (X - mean(X)) ./ std(X);
        
            % 2. Korrelationsmatrix berechnen (da std ≠ 1)
            R = corrcoef(X_norm);  % oder: R = (1 / size(X,1)) * (X_norm' * X_norm);
        
            % 3. Eigenwerte berechnen
            lambda = eig(R);
        end


        function scaleInputs(obj)
            X = obj.X_current;
            n = size(X, 2);
            obj.mu = zeros(1, n);
            obj.sigma = ones(1, n);
        
            for j = 2:n  % skip bias term if present
                mu_j = mean(X(:, j));
                sigma_j = std(X(:, j));
                X(:, j) = (X(:, j) - mu_j) / sigma_j;
                obj.mu(j) = mu_j;
                obj.sigma(j) = sigma_j;
            end
        
            obj.X_current = X;
        end


        function g = gradient(obj)
            m = size(obj.X_current, 1);
            X_design = [ones(m, 1), obj.X_current];
            g = (1 / m) * (X_design' * (X_design * obj.B - obj.y));
        end

        function cost_history = gradientDescent(obj, alpha, num_iter)
            X_design = [ones(size(obj.X_current, 1), 1), obj.X_current];
            obj.B = zeros(size(X_design, 2), 1);
            cost_history = zeros(num_iter, 1);
            for k = 1:num_iter
                g = obj.gradient();
                obj.B = obj.B - alpha * g;
                cost_history(k) = obj.cost();
            end
        end

        function B_rescaled = rescaleParameters(obj)
            B_scaled = obj.B;
            mu = obj.mu;
            sigma = obj.sigma;
            n = length(B_scaled);
            B_rescaled = zeros(n, 1);
            for i = 2:n
                B_rescaled(i) = B_scaled(i) / sigma(i);
            end
            correction = 0;
            for i = 2:n
                correction = correction + B_scaled(i) * mu(i) / sigma(i);
            end
            B_rescaled(1) = B_scaled(1) - correction;
        end

        function new_model = createPolyFeatures(obj, index, powers)
            x = obj.X(:, index);
            n = size(x, 1);
            num_terms = length(powers);
            X_poly = zeros(n, num_terms);
            for j = 1:num_terms
                X_poly(:, j) = x.^powers(j);
            end
            new_model = regression.LinearModel(X_poly, obj.y);
        end

        function summary(obj, X_val, y_val)
            % Zeigt Informationen über das aktuelle Modell:
            % - Kosten
            % - Parametervektor B
            % - Bestimmtheitsmaß R² auf Validierungsdaten
        
            if isempty(obj.B)
                error('Bitte zuerst fit() aufrufen, um das Modell zu trainieren.');
            end
        
            % Kosten für Trainingsdaten berechnen
            C = obj.cost();
            fprintf('\n--- Modellinformationen ---\n');
            fprintf('Kosten (Training): %.4f\n', C);
        
            % Parameter ausgeben
            disp('Parametervektor B:');
            disp(obj.B);
        
            % R² auf Validierungs-/Testdaten
            if nargin == 3 && ~isempty(X_val) && ~isempty(y_val)
                R2_val = obj.r2(X_val, y_val);
                fprintf('Bestimmtheitsmaß R² (auf übergebenen Daten): %.4f\n', R2_val);
            end
        end

    end
end
