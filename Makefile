mtlb:
	@echo "Running MATLAB path: $(MATLAB_PATH)"
	@echo "Running Project path: $(PROJECT_DIR)"
	$(MATLAB_PATH) -nodesktop -nosplash -sd "$(PROJECT_DIR)/" -r "run('$(run).m');"
