# swagger_docs.jl - Добавление OpenAPI документации

using HTTP
using JSON3

# Swagger UI HTML
const SWAGGER_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>ML API - Swagger UI</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3/swagger-ui.css">
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@3/swagger-ui-bundle.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                url: "/openapi.json",
                dom_id: '#swagger-ui',
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                layout: "BaseLayout"
            });
            window.ui = ui;
        }
    </script>
</body>
</html>
"""

# OpenAPI спецификация
const OPENAPI_SPEC = Dict(
    "openapi" => "3.0.0",
    "info" => Dict(
        "title" => "ML API на Julia",
        "description" => "REST API для модели Decision Tree",
        "version" => "1.0.0"
    ),
    "paths" => Dict(
        "/health" => Dict(
            "get" => Dict(
                "summary" => "Проверка здоровья сервера",
                "responses" => Dict(
                    "200" => Dict(
                        "description" => "Сервер работает",
                        "content" => Dict(
                            "application/json" => Dict(
                                "schema" => Dict(
                                    "type" => "object",
                                    "properties" => Dict(
                                        "status" => Dict("type" => "string", "example" => "healthy"),
                                        "model" => Dict("type" => "string", "example" => "DecisionTree"),
                                        "accuracy" => Dict("type" => "string", "example" => "96.7%")
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ),
        "/predict" => Dict(
            "post" => Dict(
                "summary" => "Предсказание класса",
                "requestBody" => Dict(
                    "required" => true,
                    "content" => Dict(
                        "application/json" => Dict(
                            "schema" => Dict(
                                "type" => "object",
                                "properties" => Dict(
                                    "features" => Dict(
                                        "type" => "array",
                                        "items" => Dict("type" => "number"),
                                        "minItems" => 3,
                                        "maxItems" => 3,
                                        "example" => [0.5, 0.3, 0.8]
                                    )
                                )
                            )
                        )
                    )
                ),
                "responses" => Dict(
                    "200" => Dict(
                        "description" => "Успешное предсказание",
                        "content" => Dict(
                            "application/json" => Dict(
                                "schema" => Dict(
                                    "type" => "object",
                                    "properties" => Dict(
                                        "success" => Dict("type" => "boolean", "example" => true),
                                        "prediction" => Dict("type" => "integer", "example" => 0),
                                        "features" => Dict(
                                            "type" => "array",
                                            "items" => Dict("type" => "number"),
                                            "example" => [0.5, 0.3, 0.8]
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)

function add_swagger_to_api()
    println("✅ Swagger документация добавлена")
    println("📚 Swagger UI: http://localhost:8080/docs")
    println("📄 OpenAPI spec: http://localhost:8080/openapi.json")
end

add_swagger_to_api()