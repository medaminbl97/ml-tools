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
        lambda      % Regularisierungsstärke
        reg     % 'L1' oder 'L2'

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
            obj.lambda = 0;
            obj.reg = "L2";
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

        function [C, gC] = cost(obj, beta)
            if nargin > 1
                obj.B = beta;
            end
            m = size(obj.X_current, 1);
            X_design = [ones(m, 1), obj.X_current];
            e = obj.y - X_design * obj.B;
            obj.h = X_design * obj.B;
            C = (1 / (2 * m)) * (e' * e) + obj.getCostPenalty();
            if nargout > 1
                gC = (1 / m) * (X_design' * (X_design * obj.B - obj.y)) + obj.getGradPenalty();
            end
        end


        function g = gradient(obj)
            m = size(obj.X_current, 1);
            X_design = [ones(m, 1), obj.X_current];
            g = (1 / m) * (X_design' * (X_design * obj.B - obj.y)) + obj.getGradPenalty();
        end

        function cost_history = gradientDescent(obj, alpha, num_iter)
            X_design = [ones(size(obj.X_current, 1), 1), obj.X_current];
            obj.scaleInputs()
            obj.B = zeros(size(X_design, 2), 1);
            cost_history = zeros(num_iter, 1);
            for k = 1:num_iter
                [c,gc] = obj.cost();
                obj.B = obj.B - alpha * gc;
                cost_history(k) = c;
            end
            obj.B = obj.rescaleParameters();
        end

        function C = trainFminunc(obj, maxIter, log)
            if nargin < 3
                log = false;
            end
        
            options = optimoptions('fminunc', ...
                'Algorithm', 'trust-region', ...
                'GradObj', 'on', ...
                'MaxIter', maxIter);
        
            initial_beta = zeros(size(obj.X_current, 2) + 1, 1);
            costFunc = @(b) obj.cost(b);  % closure mit optionalem beta
        
            [optB, Cval] = fminunc(costFunc, initial_beta, options);
        
            obj.B = optB;
            C = Cval;
            if log
                fprintf('Kostenfunktion beim Optimalwert: %f\n', C);
                fprintf('Optimales beta:\n');
                fprintf(' %f\n', obj.B);
            end
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

        function score = r2(obj, X_val, y_val)
            if isempty(obj.B)
                error('Model must be fitted before calling r2().');
            end
            X_val = [y_val.^0, X_val(:, obj.featureInds)];  % Bias-Spalte hinzufügen
            y_est = X_val * obj.B;
        
            % Nutze statische Methode
            score = regression.LinearModel.r2score(y_val, y_est);
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
        function B_rescaled = rescaleParameters(obj)
            B_scaled = obj.B;
            mu = obj.mu;
            sigma = obj.sigma;
            n = length(B_scaled);
            B_rescaled = zeros(n, 1);
            for i = 2:n
                B_rescaled(i) = B_scaled(i) / sigma(i-1);
            end
            correction = 0;
            for i = 2:n
                correction = correction + B_scaled(i) * mu(i-1) / sigma(i-1);
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
        function penalty = getCostPenalty(obj)
            if isempty(obj.lambda) || obj.lambda == 0
                penalty = 0;
                return;
            end

            m = size(obj.X_current,1);
        
            switch upper(obj.reg)
                case 'L2'
                    penalty = (obj.lambda / 2 / m) * sum(obj.B(2:end).^2); % ohne Bias
                case 'L1'
                    penalty = obj.lambda * sum(abs(obj.B(2:end))); % ohne Bias
                otherwise
                    error('Unbekannter Regularisierungstyp: %s', obj.reg);
            end
        end

        function penalty = getGradPenalty(obj)
            if isempty(obj.lambda) || obj.lambda == 0
                penalty = 0;
                return;
            end

            m = size(obj.X_current,1);
        
            switch upper(obj.reg)
                case 'L2'
                    penalty = (obj.lambda / m) .* [0;obj.B(2:end)]; % ohne Bias
                case 'L1'
                    penalty = (obj.lambda / m) .* sign([0;obj.B(2:end)]); % ohne Bias
                otherwise
                    error('Unbekannter Regularisierungstyp: %s', obj.reg);
            end
        end
    end
    

    methods (Static)
    function score = r2score(y_val, y_est)
        % Statische Methode zur Berechnung von R²
        ss_res = sum((y_val - y_est).^2);
        ss_tot = sum((y_val - mean(y_val)).^2);
        score = 1 - (ss_res / ss_tot);
    end
    function [x_plot, y_est] = predictLinearRange(beta, x_raw, N)
        % predictLinearRange: erzeugt Vorhersagekurve für lineares Modell
        %
        % INPUT:
        %   beta    - [beta0; beta1] (2x1) Parametervektor
        %   x_raw   - Eingabedaten (z. B. X(:,2)), zur Bereichsbestimmung
        %   N       - Anzahl der Punkte für die Kurve (optional, Default = 100)
        %
        % OUTPUT:
        %   x_plot  - gleichmäßig verteilte x-Werte (Nx1)
        %   y_est   - Vorhersage y = beta0 + beta1 * x_plot (Nx1)
    
        if nargin < 3
            N = 100;
        end
    
        % Sicherheits-Checks
        beta = beta(:);           % stelle sicher, dass beta Spaltenvektor ist
        x_raw = x_raw(:);         % x_raw ebenfalls Spaltenvektor
        assert(length(beta) == 2, 'Beta muss 2x1 sein: [bias; gewicht]');
    
        % Bereich bestimmen
        x_min = min(x_raw);
        x_max = max(x_raw);
    
        % gleichmäßige x-Werte erzeugen
        x_plot = linspace(x_min, x_max, N)';      % Spaltenvektor
        X_design = [ones(N, 1), x_plot];          % Designmatrix
        y_est = X_design * beta;                 % Vorhersage
    
        % Optional: y als Spalte erzwingen
        y_est = y_est(:);
    end



    end
end
