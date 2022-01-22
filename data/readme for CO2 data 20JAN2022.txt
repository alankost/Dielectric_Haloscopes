csv
---

The fields output for this profile are listed in order below with format strings, units and description the following information: :

nu
--
Data type: float
Units: cm-1
Description: Transition wavenumber

sw
--
Data type: float
Units: cm-1/(molec.cm-2)
Description: Line intensity, multiplied by isotopologue abundance, at T = 296 K

gamma_self
----------
Data type: float
Units: cm-1.atm-1
Description: Self-broadened HWHM at 1 atm pressure and 296 K

n_air
-----
Data type: float
Units: [dimensionless]
Description: Temperature exponent for the air-broadened HWHM

delta_self
----------
Data type: float
Units: cm-1.atm-1
Description: Self-induced pressure shift, referred to p=1 atm

elower
------
Data type: float
Units: cm-1
Description: Lower-state energy

gp
--
Data type: int
Units: [dimensionless]
Description: Upper state degeneracy

gpp
---
Data type: int
Units: [dimensionless]
Description: Lower state degeneracy

molec_id
--------
Data type: int
Units: [dimensionless]
Description: The HITRAN integer ID for this molecule in all its isotopologue forms

local_iso_id
------------
Data type: int
Units: [dimensionless]
Description: Integer ID of a particular Isotopologue, unique only to a given molecule, in order or abundance (1 = most abundant)

nu-err
------
Data type: int
Units: [dimensionless]
Description: HITRAN uncertainty code for nu

sw-err
------
Data type: int
Units: [dimensionless]
Description: HITRAN uncertainty code for sw

gamma_self-err
--------------
Data type: int
Units: [dimensionless]
Description: HITRAN uncertainty code for gamma_self

n_air-err
---------
Data type: int
Units: [dimensionless]
Description: HITRAN uncertainty code for n_air

delta_self-err
--------------
Data type: int
Units: [dimensionless]
Description: HITRAN uncertainty code for delta_self

nu-ref
------
Data type: int
Units: [dimensionless]
Description: Source (reference) ID for nu

sw-ref
------
Data type: int
Units: [dimensionless]
Description: Source (reference) ID for sw

gamma_self-ref
--------------
Data type: int
Units: [dimensionless]
Description: Source (reference) ID for gamma_self

n_air-ref
---------
Data type: int
Units: [dimensionless]
Description: Source (reference) ID for n_air

delta_self-ref
--------------
Data type: int
Units: [dimensionless]
Description: Source (reference) ID for delta_self
