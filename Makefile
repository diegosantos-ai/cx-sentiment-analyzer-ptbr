# Makefile para DevOps Workspace Central

# ==============================================================================
# Variável Global de Plataforma (Platform Engineering)
# Permite que este Makefile seja copiado via 'adopt_governance' para qualquer
# outro repositório cliente e continue ativando as automações centrais.
# ==============================================================================
DEV_WORKSPACE ?= $(HOME)/dev-workspace

.PHONY: help setup lint test update dbg

# Cores para output
CYAN := \033[36m
RESET := \033[0m

help: ## Mostra esta mensagem de ajuda
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(CYAN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Bootstrapping inicial da nova máquina (instala pacotes, links de dotfiles)
	@echo "Iniciando setup da máquina..."
	@bash $(DEV_WORKSPACE)/ansible/scripts/setup-machine.sh

setup-agents: ## Provisiona orquestradores e IAs centralizadas via pipx
	@echo "Iniciando setup do motor de Agentes IA..."
	@bash $(DEV_WORKSPACE)/gestao-centralizada-agents/scripts/setup-agents.sh

lint: ## Roda linters em todo o repositório local (Shell, Terraform, Python, Markdown)
	@echo "Executando pre-commit hooks..."
	PATH="$$HOME/.local/bin:$$PATH" pre-commit run --all-files

update: ## Atualiza repositórios, dotfiles e pacotes locais
	@echo "Atualizando ambiente local..."
	@git pull origin main
	# Adicionar comandos de update do sistema operacional aqui

# --- Sanidade e Setup Operacional ---
.PHONY: env-check

env-check: ## Rodar verificação rápida de sanidade do sistema para o dia a dia
	@bash $(DEV_WORKSPACE)/sanidade-ambiente/scripts/daily-check.sh

morning: ## Roda a rotina matinal (abre arquivos de playbook e executa o check)
	@bash $(DEV_WORKSPACE)/rotina-devops/scripts/open_devops_routine.sh

audit: ## Dispara script completo de auditoria do sistema operacional e CLI libs
	@bash $(DEV_WORKSPACE)/sanidade-ambiente/scripts/env-audit.sh

test-skills: ## Confirma se o Servidor MCP das Skills compila sem erros
	@echo "Testando build do servidor MCP de Skills..."
	@cd $(DEV_WORKSPACE)/gestao-centralizada-agents/skills-mcp && npm install && npm run build
	@echo "✅ Servidor MCP validado e pronto para consumo!"

# --- Rotina DevOps (Worklog) ---
.PHONY: day-start log day-close week-close

day-start: ## Inicia o worklog do dia e abre no VS Code
	@bash $(DEV_WORKSPACE)/rotina-devops/scripts/worklog-start.sh

log: ## Adiciona registro no log. (Sem args = interativo)
	@bash $(DEV_WORKSPACE)/rotina-devops/scripts/worklog-add.sh $(ARGS)

day-close: ## Faz o fechamento e resumo do dia atual
	@bash $(DEV_WORKSPACE)/rotina-devops/scripts/worklog-close.sh

week-close: ## Gera relatório consolidado da semana
	@bash $(DEV_WORKSPACE)/rotina-devops/scripts/worklog-weekly.sh
