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
        function obj = PCA(X)
            % Konstruktor
            obj.X = X;
            obj.centerAndScale();
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

        function k_auto = chooseK(obj, threshold)
            % Wählt automatisch K, sodass Schwelle erreicht wird (z.B. 0.95)
            eigenvalues = diag(obj.L);
            cumVar = cumsum(eigenvalues) / sum(eigenvalues);
            k_auto = find(cumVar >= threshold, 1);
        end
    end
end
