classdef KNNClassifier < handle
    properties
        X_train     % Trainingsdaten
        y_train     % Trainingslabels
        K = 3       % Anzahl der nächsten Nachbarn (Default = 3)
    end

    methods
        function obj = KNNClassifier(X, y, K)
            % Konstruktor
            obj.X_train = X;
            obj.y_train = y;
            if nargin > 2
                obj.K = K;
            end
        end

        function setK(obj, newK)
            % Zum nachträglichen Setzen von K
            obj.K = newK;
        end

        function y_pred = predict(obj, X_val)
            % Führt KNN-Klassifikation durch auf Validierungsdaten
            n_val = size(X_val, 1);
            y_pred = zeros(n_val, 1);

            for i = 1:n_val
                x = X_val(i, :);

                % (a) Euklidische Distanzen
                dists = sqrt(sum((obj.X_train - x).^2, 2));

                % (b) Sortieren und K nächste Nachbarn
                [~, sorted_idx] = sort(dists);
                knn_idx = sorted_idx(1:obj.K);

                % (c) K Nachbarlabels
                knn_labels = obj.y_train(knn_idx);

                % (d) Mehrheit bestimmen
                y_pred(i) = mode(knn_labels);
            end
        end

        function acc = accuracy(obj, X_val, y_val)
            % Erkennungsrate berechnen
            y_hat = obj.predict(X_val);
            acc = mean(y_hat == y_val);
        end
    end
end
