#################################
# Intervention Model variables #
#################################

# noinspection PyPep8Naming
class __OralIronIntervention:
    MODEL_NAME: str = "oral_iron_intervention"
    NO_TREATMENT: str = "no_treatment"
    IFA: str = "ifa"
    MMS: str = "mms"


ORAL_IRON_INTERVENTION = __OralIronIntervention()

# noinspection PyPep8Naming
class __IVIronIntervention:
    MODEL_NAME: str = "iv_iron_intervention"
    UNCOVERED: str = "uncovered"
    COVERED: str = "covered"


IV_IRON_INTERVENTION = __IVIronIntervention()
