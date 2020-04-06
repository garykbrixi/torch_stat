def pearsonr(x,y):
  """
  Based on scipy pearsonr https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
  Calculates the pearson coefficient

  Measures linear relationship between two datasets. Requires normal distribution.
  """
  mx_val = torch.mean(x)
  my_val = torch.mean(y)
  mx, my = x - mx_val, y - my_val
  r_num = torch.dot(mx,my)
  r_den = torch.norm(mx, 2) * torch.norm(my, 2)
  r = r_num / r_den

  return r

def corrcoef(x):
  c = cov_row(x)

  diag = torch.diag(c)
  stddev = torch.pow(diag, 0.5).expand_as(c)
  c = c / stddev
  c = c / (stddev.t())

  c = torch.clamp(c, -1.0, 1.0)
  return c

def cov_row(x):
  mx_val = torch.mean(x, 1)
  mx = x - mx_val.expand_as(x)
  c = torch.mm(mx, mx.T)
  c = c / (x.size(1) - 1)
  return c


def cov(x, rowvar = False):
  if not rowvar:
    x = x.t()
  f = 1.0 / (x.size(1) - 1)
  x -= torch.mean(x, dim=1, keepdim=True)
  xt = x.t()
  c = f * x.matmul(xt).squeeze()
  return c


def PCA(data, dims = 2):
  """
  based on numpy function:
  https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
  """
  n = data.shape
  mx = torch.mean(data, dim=0)

  x = data - mx
  R = cov(x, False)

  evals, evecs = torch.symeig(R_t, eigenvectors = True, upper = True)

  idx = torch.argsort(evals, descending = True)
  evecs = evecs[:,idx]
  evals = evals[idx]

  evecs = evecs[:, :dims]

  return torch.mm(evecs.T, data.T).T, evals, evecs
