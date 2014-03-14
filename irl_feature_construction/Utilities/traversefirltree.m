function [leaf, v, str] = traversefirltree(tree, s, str, feature_data, nc1)

% Check if this is a leaf.
if tree.type == 0,
    % Return result.
    leaf = tree.index;
    v    = mean(tree.mean);
else
    % Recurse.
    if feature_data.splittable(s, tree.test) == 0,
        branch = tree.ltTree;
        d = ceil(tree.test/nc1);
        i = mod(tree.test - 1, nc1) + 1;
%         x = sprintf('%s$d(%d) \\geq  %2d$ & ', str, i, d);
        x = sprintf('%sd(%d) >=  %2d & ', str, i, d);
    else
        branch = tree.gtTree;
        d = ceil(tree.test/nc1);
        i = mod(tree.test - 1, nc1) + 1;
%         x = sprintf('%s$d(%d) < %2d$ & ', str, i, d);
        x = sprintf('%sd(%d) < %2d & ', str, i, d);
    end
    [leaf, v, str] = traversefirltree(branch, s, x, feature_data, nc1);
end