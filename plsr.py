from sklearn.utils.validation import check_is_fitted, check_array

def svd_flip(u, v, u_based_decision=True):
    u = u.view(u.size()[0],1)
    v = v.view(v.size()[0],1)
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = torch.argmax(torch.abs(u), axis=0)
        print(max_abs_cols)
        signs = torch.sign(u[list(max_abs_cols), range(u[0].size()[0])])
        u *= signs
        v *= signs.unsqueeze(1)
    else:
        # rows of v, columns of u
        max_abs_rows = torch.argmax(torch.abs(v), axis=1)
        signs = torch.sign(v[range(list(v.size())[0]), max_abs_rows])
        u *= signs
        v *= signs.unsqueeze(1)
    u = u.flatten()
    v = v.flatten()
    return u, v

def _nipals_twoblocks_inner_loop(X, Y, mode="A", max_iter=500, tol=1e-06,
                                 norm_y_weights=False):
    """Inner loop of the iterative NIPALS algorithm.
    Provides an alternative to the svd(X'Y); returns the first left and right
    singular vectors of X'Y.  See PLS for the meaning of the parameters.  It is
    similar to the Power method for determining the eigenvectors and
    eigenvalues of a X'Y.
    """

    for col in Y.T:
        if torch.any(torch.abs(col) > torch.finfo(torch.double).eps):

            y_score = col.detach().view(col.size())

            break

    x_weights_old = 0
    ite = 1
    X_pinv = Y_pinv = None
    eps = torch.finfo(X.dtype).eps

    if mode == "B":
        # Uses condition from scipy<1.3 in pinv2 which was changed in
        # https://github.com/scipy/scipy/pull/10067. In scipy 1.3, the
        # condition was changed to depend on the largest singular value
        X_t = X.dtype.char.lower()
        Y_t = Y.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}

        cond_X = factor[X_t] * eps
        cond_Y = factor[Y_t] * eps

    # Inner loop of the Wold algo.
    while True:
        # 1.1 Update u: the X weights
        if mode == "B":
            if X_pinv is None:
                # torch pinverse
                X_pinv = torch.pinverse(X, check_finite=False, cond=cond_X)
            x_weights = torch.mm(X_pinv, y_score)
        else:  # mode
            # Mode A regress each X column on y_score

            x_weights = torch.mv(X.T, y_score) / torch.dot(y_score.T, y_score)
        # If y_score only has zeros x_weights will only have zeros. In
        # this case add an epsilon to converge to a more acceptable
        # solution
        if torch.dot(x_weights.T, x_weights) < eps:
            x_weights += eps
        # 1.2 Normalize u
        x_weights /= torch.sqrt(torch.dot(x_weights.T, x_weights)) + eps
        # 1.3 Update x_score: the X latent scores
        x_score = torch.mv(X, x_weights)
        # 2.1 Update y_weights
        if mode == "B":
            if Y_pinv is None:
                # compute once pinv(Y)
                Y_pinv = torch.pinverse(Y, check_finite=False, cond=cond_Y)
            y_weights = torch.mm(Y_pinv, x_score)
        else:
            # Mode A regress each Y column on x_score
            y_weights = torch.mv(Y.T, x_score) / torch.dot(x_score.T, x_score)
        # 2.2 Normalize y_weights
        if norm_y_weights:
            y_weights /= torch.sqrt(torch.mm(y_weights.T, y_weights)) + eps
        # 2.3 Update y_score: the Y latent scores
        y_score = torch.mv(Y, y_weights) / (torch.dot(y_weights.T, y_weights) + eps)
        # y_score = np.dot(Y, y_weights) / np.dot(y_score.T, y_score) ## BUG
        x_weights_diff = x_weights - x_weights_old

        if torch.dot(x_weights_diff.T, x_weights_diff) < tol or Y.size()[1] == 1:
            break
        if ite == max_iter:
            warnings.warn('Maximum number of iterations reached',
                          ConvergenceWarning)
            break
        x_weights_old = x_weights
        ite += 1
    return x_weights, y_weights, ite

def _center_scale_xy(X, Y, scale=True):
    """ Center X, Y and scale if the scale parameter==True
    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """
    # center
    x_mean = torch.mean(X, axis=0)
    X -= x_mean
    y_mean = torch.mean(Y, axis=0)
    Y -= y_mean
    # scale
    if scale:
        x_std = torch.std(X, dim = 0)
        x_std[x_std == 0.0] = 1.0
        X = X/x_std
        y_std = torch.std(Y, dim = 0)
        y_std[y_std == 0.0] = 1.0
        Y = Y/y_std
    else:
        x_std = torch.ones(X.size()[1])
        y_std = torch.ones(Y.size()[1])
    return X, Y, x_mean, y_mean, x_std, y_std

class PLS:
    def __init__(self, n_components=2, *, scale=True,
                 deflation_mode="regression",
                 mode="A", algorithm="nipals", norm_y_weights=False,
                 max_iter=500, tol=1e-06, copy=True):
        self.n_components = n_components
        self.deflation_mode = deflation_mode
        self.mode = mode
        self.norm_y_weights = norm_y_weights
        self.scale = scale
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
        self.scale = scale

    def fit(self, X, Y):
        """Fit model to data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
        Y : array-like of shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
        """
        X = X.clone().detach()
        Y = Y.clone().detach()
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n = X.size()[0]
        p = X.size()[1]
        q = Y.size()[1]

        # if self.n_components < 1 or self.n_components > p:
        #     raise ValueError('Invalid number of components: %d' %
        #                      self.n_components)
        # if self.algorithm not in ("svd", "nipals"):
        #     raise ValueError("Got algorithm %s when only 'svd' "
        #                      "and 'nipals' are known" % self.algorithm)
        # if self.algorithm == "svd" and self.mode == "B":
        #     raise ValueError('Incompatible configuration: mode B is not '
        #                      'implemented with svd algorithm')
        # if self.deflation_mode not in ["canonical", "regression"]:
        #     raise ValueError('The deflation mode is unknown')
        # Scale (in place)
        X, Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = (
            _center_scale_xy(X, Y, self.scale))
        # Residuals (deflated) matrices
        Xk = X
        Yk = Y

        # Results matrices
        self.x_scores_ = torch.zeros((n, self.n_components))
        self.y_scores_ = torch.zeros((n, self.n_components))
        self.x_weights_ = torch.zeros((p, self.n_components))
        self.y_weights_ = torch.zeros((q, self.n_components))
        self.x_loadings_ = torch.zeros((p, self.n_components))
        self.y_loadings_ = torch.zeros((q, self.n_components))
        self.n_iter_ = []

        # NIPALS algo: outer loop, over components
        Y_eps = torch.finfo(Yk.dtype).eps

        for k in range(self.n_components):
            print('Xk')
            print(Xk)
            print('Yk')
            print(Yk)
            if torch.all(torch.mm(Yk.T, Yk) < torch.finfo(torch.double).eps):
                # Yk constant
                warnings.warn('Y residual constant at iteration %s' % k)
                break
            # 1) weights estimation (inner loop)
            # -----------------------------------
            if self.algorithm == "nipals":
                # Replace columns that are all close to zero with zeros
                Yk_mask = torch.all(torch.abs(Yk) < 10 * Y_eps, axis=0)
                Yk[:, Yk_mask] = 0.0

                x_weights, y_weights, n_iter_ = \
                    _nipals_twoblocks_inner_loop(
                        X=Xk, Y=Yk, mode=self.mode, max_iter=self.max_iter,
                        tol=self.tol, norm_y_weights=self.norm_y_weights)
                self.n_iter_.append(n_iter_)

                print("self.n_iter_")
                print(self.n_iter_)
            elif self.algorithm == "svd":
                x_weights, y_weights = _svd_cross_product(X=Xk, Y=Yk)
            # Forces sign stability of x_weights and y_weights
            # Sign undeterminacy issue from svd if algorithm == "svd"
            # and from platform dependent computation if algorithm == 'nipals'

            x_weights, y_weights = svd_flip(x_weights, y_weights.T)
            y_weights = y_weights.T
            # columns of u, rows of v

            # compute scores
            x_scores = torch.mv(Xk, x_weights)

            if self.norm_y_weights:
                y_ss = 1
            else:
                y_ss = torch.dot(y_weights.T, y_weights)

            y_scores = torch.mv(Yk, y_weights) / y_ss

            # test for null variance
            if torch.dot(x_scores.T, x_scores) < torch.finfo(torch.double).eps:
                warnings.warn('X scores are null at iteration %s' % k)
                break
            # 2) Deflation (in place)
            # ----------------------
            #
            # - regress Xk's on x_score

            x_loadings = torch.mv(Xk.T, x_scores) / torch.dot(x_scores.T, x_scores)

            # - subtract rank-one approximations to obtain remainder matrix

            Xk -= x_scores[:, None] * x_loadings.T

            if self.deflation_mode == "canonical":
                # - regress Yk's on y_score, then subtract rank-one approx.
                y_loadings = (torch.mv(Yk.T, y_scores)
                              / torch.dot(y_scores.T, y_scores))
                Yk -= y_scores[:, None] * y_loadings.T
            if self.deflation_mode == "regression":
                # - regress Yk's on x_score, then subtract rank-one approx.
                y_loadings = (torch.mv(Yk.T, x_scores)
                              / torch.dot(x_scores.T, x_scores))
                Yk -= x_scores[:, None] * y_loadings.T
            # 3) Store weights, scores and loadings # Notation:
            self.x_scores_[:, k] = x_scores.view(-1)  # T
            self.y_scores_[:, k] = y_scores.view(-1)  # U
            self.x_weights_[:, k] = x_weights.view(-1)  # W
            self.y_weights_[:, k] = y_weights.view(-1)  # C
            self.x_loadings_[:, k] = x_loadings.view(-1)  # P
            self.y_loadings_[:, k] = y_loadings.view(-1)  # Q

        # Such that: X = TP' + Err and Y = UQ' + Err

        # 4) rotations from input space to transformed space (scores)
        # T = X W(P'W)^-1 = XW* (W* : p x k matrix)
        # U = Y C(Q'C)^-1 = YC* (W* : q x k matrix)
        self.x_rotations_ = torch.mm(
            self.x_weights_,
            torch.pinverse(torch.mm(self.x_loadings_.T, self.x_weights_)))
        if Y.size()[1] > 1:
            self.y_rotations_ = torch.mm(
                self.y_weights_,
                torch.pinverse(torch.mm(self.y_loadings_.T, self.y_weights_)))
        else:
            self.y_rotations_ = torch.ones(1)

        if True or self.deflation_mode == "regression":
            # Estimate regression coefficient
            # Regress Y on T
            # Y = TQ' + Err,
            # Then express in function of X
            # Y = X W(P'W)^-1Q' + Err = XB + Err
            # => B = W*Q' (p x q)

            self.coef_ = torch.mm(self.x_rotations_, self.y_loadings_.T)
            self.coef_ = self.coef_.to("cuda")
            self.y_std_ = self.y_std_.to("cuda")
            # self.coef_ = torch.mv(self.coef_, self.y_std_)
            self.coef_ = self.coef_[:, None] * self.y_std_
            self.coef_ = self.coef_[:,0,:]

        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
        Y : array-like of shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.
        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        check_is_fitted(self)
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        # Normalize
        X -= self.x_mean_
        X /= self.x_std_
        # Apply rotation
        x_scores = torch.mm(X, self.x_rotations_)
        if Y is not None:
            Y = check_array(Y, ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            Y -= self.y_mean_
            Y /= self.y_std_
            y_scores = torch.mm(Y, self.y_rotations_)
            return x_scores, y_scores

        return x_scores

    def inverse_transform(self, X):
        """Transform data back to its original space.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of pls components.
        Returns
        -------
        x_reconstructed : array-like of shape (n_samples, n_features)
        Notes
        -----
        This transformation will only be exact if n_components=n_features
        """
        check_is_fitted(self)
        X = check_array(X, dtype=FLOAT_DTYPES)
        # From pls space to original space
        X_reconstructed = torch.matmul(X, self.x_loadings_.T)

        # Denormalize
        X_reconstructed *= self.x_std_
        X_reconstructed += self.x_mean_
        return X_reconstructed

    def predict(self, X, copy=True):
        """Apply the dimension reduction learned on the train data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.
        Notes
        -----
        This call requires the estimation of a p x q matrix, which may
        be an issue in high dimensional space.
        """
        check_is_fitted(self)
        # X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        # Normalize

        X -= self.x_mean_
        X /= self.x_std_

        Ypred = torch.mm(X, self.coef_)
        return Ypred + self.y_mean_

    def fit_transform(self, X, y=None):
        """Learn and apply the dimension reduction on the train data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
        y : array-like of shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        return self.fit(X, y).transform(X, y)

    def _more_tags(self):
        return {'poor_score': True}
