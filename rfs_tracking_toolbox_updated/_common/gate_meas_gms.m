function z_gate= gate_meas_gms(z,gamma,model,m,P)

valid_idx = [];
zlength = size(z,2); if zlength==0, z_gate= []; return; end
plength = size(m,2);

for j=1:plength
    Sj= model.R + model.H*P(:,:,j)*model.H';
    nu = z - model.H * repmat(m(:,j), [1 zlength]);
    
    Vs = chol(Sj); 
    inv_sqrt_Sj= inv(Vs);
    
    dist = sum((inv_sqrt_Sj'*nu).^2);
    dist_ = diag(nu' * inv(Sj) * nu)';

    valid_idx= union(valid_idx,find( dist < gamma ));
end
z_gate = z(:,valid_idx);