function calcularResistencia() {
    let r1 = parseFloat(document.getElementById("res1").value);
    let r2 = parseFloat(document.getElementById("res2").value);
    
    if (r1 > 0 && r2 > 0) {
        let rEquivalente = (r1 * r2) / (r1 + r2);
        document.getElementById("resultado").innerText = rEquivalente.toFixed(2);
    } else {
        alert("Ingrese valores válidos para las resistencias.");
    }
}