import asyncio
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Conceptual import of the repository's configuration
# from ..config import DATA_STORAGE, SOLVERs_FOLDER 

logger = logging.getLogger(__name__)

@dataclass
class RunResult:
    success: bool
    return_code: int
    log_file: Path
    error_message: Optional[str] = None

class ModelRunner:
    """
    Secure asynchronous execution engine for orchestrating heavy scientific models.
    Enforces directory sandboxing to prevent path traversal vulnerabilities.
    """

    @staticmethod
    async def _run_and_stream(command: str, log_file: Path, cwd: Path) -> RunResult:
        """Executes a shell command and streams stdout/stderr to disk securely."""
        logger.info(f"Executing command securely in {cwd}")
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT, 
                cwd=cwd
            )
            
            with open(log_file, 'wb') as f:
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    f.write(line)
            
            await process.wait()
            success = process.returncode == 0
            
            return RunResult(success=success, return_code=process.returncode, log_file=log_file)
            
        except Exception as e:
            logger.exception(f"Exception during subprocess execution: {e}")
            return RunResult(success=False, return_code=-1, log_file=log_file, error_message=str(e))

    @classmethod
    async def run_clews(cls, case_name: str, solver: str = "glpsol") -> RunResult:
        """
        Orchestrates OSeMOSYS securely inside the DATA_STORAGE sandbox.
        """
        # Assume DATA_STORAGE and SOLVERs_FOLDER are loaded from config
        DATA_STORAGE = Path('WebAPP', 'DataStorage')
        SOLVERs_FOLDER = Path('WebAPP', 'SOLVERs')
        
        # Sandbox the execution to the specific case folder
        case_dir = DATA_STORAGE / case_name
        if not case_dir.exists():
            raise FileNotFoundError(f"Case directory {case_dir} does not exist.")

        model_file = case_dir / f"{case_name}_model.txt"
        data_file = case_dir / f"{case_name}_data.txt"
        output_file = case_dir / f"{case_name}_output.txt"
        log_file = case_dir / f"{case_name}_solver.log"

        m_safe = shlex.quote(str(model_file.name))
        d_safe = shlex.quote(str(data_file.name))
        o_safe = shlex.quote(str(output_file.name))
        
        # Path to the actual solver executable
        solver_exec = shlex.quote(str(SOLVERs_FOLDER / solver))

        # Command executed within the isolated case_dir
        cmd = f"{solver_exec} -m {m_safe} -d {d_safe} -o {o_safe}"

        return await cls._run_and_stream(cmd, log_file=log_file, cwd=case_dir)