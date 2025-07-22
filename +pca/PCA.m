classdef PCA < handle
    properties
        X               % Originaldatenmatrix
        Xmean           % Mittelwert
        Xstd            % Standardabweichung
        R               % Kovarianzmatrix
        Q               % Eigenvektoren
        L               % Eigenwerte (Diagonalform)
        Z               % Transformierte Datenmatrix (alle Komponenten)
    end

    methods
        function obj = PCA(X, centerAndScale)
            obj.X = X;

            if nargin < 2 || centerAndScale
                obj.centerAndScale();
            end

            obj.computeCovariance();
            obj.eigenDecomposition();
            obj.transform();
        end

        function centerAndScale(obj)
            % Zentrieren und Skalieren
            obj.Xmean = mean(obj.X);
            X_centered = obj.X - obj.Xmean;
            obj.Xstd = std(obj.X);
            X_scaled = X_centered ./ obj.Xstd;
            obj.X = X_scaled;
        end

        function computeCovariance(obj)
            % Kovarianzmatrix berechnen
            m = size(obj.X, 1);
            obj.R = (1 / m) * (obj.X') * obj.X;
        end

        function eigenDecomposition(obj)
            % Eigenwertzerlegung und Sortierung
            [Q_raw, L_raw] = eig(obj.R);
            [p, idx] = sort(diag(L_raw), 'descend');
            obj.L = diag(p);
            obj.Q = Q_raw(:, idx);
        end

        function transform(obj)
            % Transformation der Daten
            obj.Z = obj.X * obj.Q; % Alle Hauptkomponenten
        end

        function Z_k = getReduced(obj, k)
            % Reduzierte transformierte Matrix mit k Hauptkomponenten
            if nargin < 2
                Z_k = obj.Z;
            else
                Z_k = obj.Z(:, 1:k);
            end
        end

        function plotEigenvalues(obj)
            % Eigenwerte plotten
            figure;
            plot(diag(obj.L), '-o', 'LineWidth', 1.5);
            xlabel('Index');
            ylabel('Eigenwert');
            title('Eigenwerte (Varianz je Hauptkomponente)');
            grid on;
        end

        function lambdas = getEigenvalues(obj)
            % Gibt die Eigenwerte als Vektor zurück
            lambdas = diag(obj.L);
        end

        function [ratio, cumRatio] = explainedVariance(obj, k)
            % Gibt den Varianzanteil der ersten k Hauptkomponenten zurück
            lambdas = diag(obj.L);
            totalVar = sum(lambdas);
            cumRatio = cumsum(lambdas) / totalVar;
            ratio = cumRatio(k);
        end

        function k_auto = chooseK(obj, threshold)
            % Wählt automatisch K, sodass Schwelle erreicht wird (z.B. 0.95)
            eigenvalues = diag(obj.L);
            cumVar = cumsum(eigenvalues) / sum(eigenvalues);
            k_auto = find(cumVar >= threshold, 1);
        end

        function printEigenInfo(obj, k)
            % Gibt Eigenwerte und erklärten Varianzanteil schön formatiert aus
            lambdas = obj.getEigenvalues();
            fprintf('Eigenwerte λ1 bis λ%d:\n', length(lambdas));
            disp(lambdas');

            [r2, ~] = obj.explainedVariance(k);
            fprintf('Anteil mit den ersten %d Hauptkomponenten: %.4f = %.2f%%\n', ...
                    k, r2, r2*100);
        end

        function topFeaturesPerComponent(obj, pcN, featN)
            % Gibt für jede der ersten pcN Hauptkomponenten die wichtigsten featN Originalmerkmale aus
        
            if nargin < 3
                featN = 2; % default: top 2 features per component
            end
            if nargin < 2
                pcN = size(obj.Q, 2); % default: all principal components
            end
        
            [~, nFeatures] = size(obj.X);
            featureLabels = "x" + (1:nFeatures);
        
            fprintf('Wichtigste %d Merkmale pro Hauptkomponente (HK):\n', featN);
            for i = 1:pcN
                [~, idx] = maxk(abs(obj.Q(:, i)), featN);
                fprintf('HK %d: %s\n', i, join(featureLabels(idx), ', '));
            end
        end

    end
end
