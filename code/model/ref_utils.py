# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for reflection directions and directional encodings."""
import torch


def reflect(viewdirs, normals):
  """Reflect view directions about normals.

  The reflection of a vector v about a unit vector n is a vector u such that
  dot(v, n) = dot(u, n), and dot(u, u) = dot(v, v). The solution to these two
  equations is u = 2 dot(n, v) n - v.

  Args:
    viewdirs: [..., 3] array of view directions.
    normals: [..., 3] array of normal directions (assumed to be unit vectors).

  Returns:
    [..., 3] array of reflection directions.
  """
  return 2.0 * torch.sum(normals * viewdirs, dim=-1, keepdims=True) * normals - viewdirs


def l2_normalize(x, eps=1e-10):
  """Normalize x to unit length along last axis."""
  return x / torch.maximum(torch.norm(x,dim=-1,keepdim=True),eps*torch.ones_like(x[...,:1]))


def linear_to_srgb(linear,
                   eps=1e-10):
  """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
  srgb0 = 323 / 25 * linear
  srgb1 = (211 * torch.maximum(eps*torch.ones_like(linear), linear)**(5 / 12) - 11) / 200
  return torch.where(linear <= 0.0031308, srgb0.to(dtype=srgb1.dtype), srgb1)


def srgb_to_linear(srgb,eps=1e-10):
  """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""

  linear0 = 25 / 323 * srgb
  linear1 = torch.maximum(eps*torch.ones_like(srgb), ((200 * srgb + 11) / (211)))**(12 / 5)
  return torch.where(srgb <= 0.04045, linear0, linear1)

