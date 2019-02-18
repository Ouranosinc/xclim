# -*- coding: utf-8 -*-
from xclim import checks
from xclim import indices as _ind
from xclim.utils import Indicator, generic_frequency_analysis, generic_max
from xclim.utils import generic_seasonal_stat_return_period
from functools import partial, update_wrapper
from boltons.funcutils import FunctionBuilder


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


docstring_template = \
    """{title}
    
    {description}
    
    Parameters
    ----------
    q : xarray.DataArray
      Input streamflow [m3/s]
    {params}
    
    Returns
    -------
    xarray.DataArray
      {output}
      
    Notes
    -----
    {notes}
    """


class Streamflow(Indicator):
    required_units = 'm^3 s-1'
    units = 'm^3 s-1'
    context = 'hydro'
    standard_name = 'streamflow'

    @staticmethod
    def compute(*args, **kwds):
        pass

    def cfprobe(self, q):
        checks.check_valid(q, 'standard_name', 'streamflow')


base_flow_index = Streamflow(identifier='base_flow_index',
                             units='',
                             standard_name='',
                             compute=_ind.base_flow_index)

q_max = Streamflow(identifier='q_max',
                   compute=generic_max,
                   )


def generate(identifier, title, description, notes='', defaults={}, fixed={}):
    """Return a Streamflow instance based on the `generic_frequency_analysis` function.

    Parameters
    ----------
    identifier : str
      Function name
    title : str
      Brief description of function's intent.
    description : str
      One paragraph explanation of function's behavior.
    notes : str
      Additional information, such as the mathematical formula.
    defaults : dict
      Default values for keyword arguments of the created function.
    fixed : dict
      Fixed values passed to the `generic_frequency_analysis` function.
    """

    # TODO: create parameter lines in docstring.
    params = ""
    output = ""
    args = ['q', ] + list(defaults.keys())
    gfa_args = ["da=q", ] + \
               ["{0}={0}".format(k) for k in defaults.keys()] + \
               ["{0}={1}".format(k, v) for (k, v) in fixed.items()]

    body = """from xclim.utils import generic_frequency_analysis as gfa\nreturn gfa({})""".format(', '.join(gfa_args))

    f = FunctionBuilder(name=identifier.lower(),
                        doc=docstring_template.format(**{'identifier': identifier,
                                                         'title': title,
                                                         'description': description,
                                                         'params': params,
                                                         'output': output,
                                                         'notes': notes}),
                        body=body,
                        args=args,
                        defaults=tuple(defaults.values())
                        )

    print(body)
    return Streamflow(identifier=identifier,
                      title=title,
                      description=description,
                      compute=f.get_func())


def generate2(identifier, title, description, notes='', defaults={}, fixed={}):
    """Return a Streamflow instance based on the `generic_frequency_analysis` function.

    Parameters
    ----------
    identifier : str
      Function name
    title : str
      Brief description of function's intent.
    description : str
      One paragraph explanation of function's behavior.
    notes : str
      Additional information, such as the mathematical formula.
    defaults : dict
      Default values for keyword arguments of the created function.
    fixed : dict
      Fixed values passed to the `generic_frequency_analysis` function.
    """

    # TODO: create parameter lines in docstring.
    params = ""
    output = ""
    args = ['q', ] + list(defaults.keys())
    gfa_args = ["da=q", ] + \
               ["{0}={0}".format(k) for k in defaults.keys()] + \
               ["{0}={1}".format(k, v) for (k, v) in fixed.items()]

    body = """from xclim.utils import generic_seasonal_stat_return_period as gfa2\nreturn gfa2({})""".format(
        ', '.join(gfa_args))

    f = FunctionBuilder(name=identifier.lower(),
                        doc=docstring_template.format(**{'identifier': identifier,
                                                         'title': title,
                                                         'description': description,
                                                         'params': params,
                                                         'output': output,
                                                         'notes': notes}),
                        body=body,
                        args=args,
                        defaults=tuple(defaults.values())
                        )

    print(body)
    return Streamflow(identifier=identifier,
                      title=title,
                      description=description,
                      compute=f.get_func())


# What I'm doing here is taking a generic function and assigning new defaults to it (for freq, t, dist, and mode).
# You still can override these defaults when calling the function, which may not be great. I think some defaults
# should be frozen (t, mode) while others could be left free (dist, freq).
q1max2sp = generate(identifier='Q1max2Sp',
                    title='2-year return period maximum spring flood peak',
                    description='Annual maximum [max] daily flow [Q1] of 2-year recurrence [2] in spring [Sp]',
                    defaults=dict(freq='QS-MAR', dist='gumbel_r'),
                    fixed=dict(t=2, mode="'high'"))

q1max2sp_new = generate2(identifier='Q1max2Sp',
                        title='2-year return period maximum spring flood peak',
                        description='Annual maximum [max] daily flow [Q1] of 2-year recurrence [2] in spring [Sp]',
                        defaults=dict(dist='gumbel_r'),
                        fixed=dict(t=2, mode="'high'", seasons="'MAM'", window=1, freq="'AS-Dec'"))
