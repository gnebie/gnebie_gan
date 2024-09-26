import os
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("gnebie_gan")
logger.setLevel(logging.DEBUG)

# logging.config.fileConfig('logging.conf')


DEBUG_V = 7
logging.addLevelName(DEBUG_V, "DEBUGV")
def debugv(self, message, *args, **kws):
    if self.isEnabledFor(DEBUG_V):
        # Yes, logger takes its '*args' as 'args'.
        self._log(DEBUG_V, message, args, **kws) 
logging.Logger.debugv = debugv

NOTICE = 15 
logging.addLevelName(NOTICE, "NOTICE")
def notice(self, message, *args, **kws):
    if self.isEnabledFor(NOTICE):
        # Yes, logger takes its '*args' as 'args'.
        self._log(NOTICE, message, args, **kws) 
logging.Logger.notice = notice

TRACE = 5
logging.addLevelName(TRACE, "TRACE")
def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE):
        # Yes, logger takes its '*args' as 'args'.
        self._log(TRACE, message, args, **kws) 
logging.Logger.trace = trace


switcher = {
    "trace":TRACE,
    "debugv":DEBUG_V,
    "debug":logging.DEBUG,
    "notice":NOTICE,
    "info":logging.INFO,
    "warning":logging.WARNING,
    "error":logging.ERROR,
    "critical":logging.CRITICAL,
}


def test_logs():
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')

def setup_file_log(flags):
    level = flags.loglevel
    log_level = switcher.get(level, logging.INFO)

    log_file = os.path.join(flags.out, flags.logfile)
    ch = RotatingFileHandler(log_file, maxBytes=10000)
    ch.setLevel(log_level)

    if False and (logging.DEBUG == log_level):
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(pathname)s - %(funcName)s l.%(lineno)d - %(levelname)s: %(message)s')
    else:
        file_formatter = logging.Formatter('%(asctime)s - %(module)spy - %(funcName)s l.%(lineno)d - %(levelname)s: %(message)s')
 

    ch.setFormatter(file_formatter)

    logger.addHandler(ch)

def setup_console_log(flags):
    console_formatter = logging.Formatter('[%(levelname)s]: - %(message)s')
    ch = logging.StreamHandler()
    if (flags.quiet):
        ch.setLevel(logging.CRITICAL)
    elif (flags.verbosity == 0):
        ch.setLevel(logging.WARNING)
    elif (flags.verbosity == 1):
        ch.setLevel(logging.INFO)
    elif (flags.verbosity == 2):
        ch.setLevel(logging.NOTICE)
    else:
        ch.setLevel(logging.DEBUG)
    ch.setFormatter(console_formatter)

    logger.addHandler(ch)
    
def setup_log(flags):
    setup_file_log(flags)
    setup_console_log(flags)
    logger.info("File Log level set : %s", flags.loglevel)
    logger.debug("arguments given : %s", flags)